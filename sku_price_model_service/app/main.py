import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Dict, List, Any
from preprocessing import preprocessing
from demand_predictor import DemandPredictor
from price_optimizer import optimize_price
from dotenv import load_dotenv

load_dotenv()

app: FastAPI = FastAPI()

predictor: DemandPredictor = DemandPredictor()


@app.post("/invocation")
async def create_upload_file(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format! Only .csv!")

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    data: pd.DataFrame = pd.read_csv(file.filename)
    os.remove(file.filename)

    processed_data: pd.DataFrame = preprocessing(data)

    predictions: np.ndarray = predictor.predict(processed_data)

    data['num_purchases'] = predictions

    return data.to_dict(orient="records")


@app.post("/optimize_price")
async def optimize_price_endpoint(file: UploadFile = File(...)) -> List[Dict[str, Any]]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format! Only .csv!")

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    data: pd.DataFrame = pd.read_csv(file.filename)
    os.remove(file.filename)

    dates = data['dates'].to_list()

    processed_data: pd.DataFrame = preprocessing(data)

    optimization_results_df: pd.DataFrame = optimize_price(processed_data, predictor)

    optimization_results = optimization_results_df.to_dict(orient="records")

    for i, result in enumerate(optimization_results):
        result['dates'] = dates[i]

    return optimization_results
