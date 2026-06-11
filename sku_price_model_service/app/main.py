from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from demand_predictor import DemandPredictor
from preprocessing import preprocessing
from price_optimizer import optimize_price

app: FastAPI = FastAPI(title="Dynamic Pricing API")
_predictor: DemandPredictor | None = None


def get_predictor() -> DemandPredictor:
    global _predictor

    if _predictor is None:
        _predictor = DemandPredictor()

    return _predictor


async def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .csv is supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        return pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=400, detail="Failed to parse uploaded CSV file.") from exc


def load_predictor_or_503() -> DemandPredictor:
    try:
        return get_predictor()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Demand model is unavailable: {exc}",
        ) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invocation")
async def create_upload_file(file: UploadFile = File(...)) -> list[dict[str, Any]]:
    data = await read_csv_upload(file)

    try:
        processed_data = preprocessing(data)
        predictions = load_predictor_or_503().predict(processed_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data["num_purchases"] = np.asarray(predictions, dtype=float)

    return data.to_dict(orient="records")


@app.post("/optimize_price")
async def optimize_price_endpoint(file: UploadFile = File(...)) -> list[dict[str, Any]]:
    data = await read_csv_upload(file)

    try:
        dates = data["dates"].astype(str).to_list()
        processed_data = preprocessing(data)
        optimization_results_df = optimize_price(processed_data, load_predictor_or_503())
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing required column: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    optimization_results = optimization_results_df.to_dict(orient="records")
    for idx, result in enumerate(optimization_results):
        result["dates"] = dates[idx]

    return optimization_results
