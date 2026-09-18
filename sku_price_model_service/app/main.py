"""HTTP API: demand prediction and price optimisation for uploaded CSV files."""

import io
import logging
import threading
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.exc import SQLAlchemyError

from .config import MODEL_ALIAS, MODEL_NAME
from .db import ReferenceData, get_engine, load_reference_data
from .demand_predictor import DemandPredictor
from .preprocessing import build_features
from .price_optimizer import optimize_prices

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Pricing API",
    description="Demand forecast and price optimisation for SKUs. Upload a CSV with dates, SKU, price_per_sku.",
    version="1.0.0",
)

_predictor: DemandPredictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> DemandPredictor:
    """Load the model behind MODEL_NAME@MODEL_ALIAS on first use and keep it.

    Restart the API after training a new model to pick it up.
    """
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            try:
                _predictor = DemandPredictor.from_registry()
            except Exception as exc:  # MLflow raises many unrelated exception types
                logger.warning("Demand model is unavailable: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"Demand model {MODEL_NAME}@{MODEL_ALIAS} is unavailable, "
                        "train it with `docker compose run --rm trainer`."
                    ),
                ) from exc
        return _predictor


def get_reference_data() -> ReferenceData:
    try:
        return load_reference_data(get_engine())
    except SQLAlchemyError as exc:
        logger.warning("Reference tables are unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Reference tables promo, sku_dict and prices are unavailable, "
                "load them with `docker compose run --rm seed`."
            ),
        ) from exc


def read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .csv is supported.")

    content = file.file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        return pd.read_csv(io.BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse the CSV file: {exc}") from exc


def json_records(frame: pd.DataFrame) -> Response:
    # pandas serialises NaN as null and handles numpy scalars.
    return Response(frame.to_json(orient="records", date_format="iso"), media_type="application/json")


@app.get("/health")
def health() -> dict[str, str | None]:
    return {"status": "ok", "model_version": _predictor.version if _predictor else None}


@app.post("/invocation")
def predict_demand(
    file: Annotated[UploadFile, File()],
    refs: Annotated[ReferenceData, Depends(get_reference_data)],
    predictor: Annotated[DemandPredictor, Depends(get_predictor)],
) -> Response:
    """Return the uploaded rows with the predicted demand in `num_purchases`."""
    data = read_csv_upload(file)
    try:
        features = build_features(data, refs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = data.copy()
    result["num_purchases"] = predictor.predict(features)
    return json_records(result)


@app.post("/optimize_price")
def optimize_price(
    file: Annotated[UploadFile, File()],
    refs: Annotated[ReferenceData, Depends(get_reference_data)],
    predictor: Annotated[DemandPredictor, Depends(get_predictor)],
) -> Response:
    """Return the optimal price per row next to the base scenario at the uploaded price."""
    data = read_csv_upload(file)
    try:
        features = build_features(data, refs)
        costs = refs.costs_for(features["SKU"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = optimize_prices(features, costs, predictor)
    result.insert(0, "dates", data["dates"].astype(str))
    return json_records(result)
