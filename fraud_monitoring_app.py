import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.predict import FraudPredictor
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)

logger = logging.getLogger(__name__)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FraudPredictor...")
    ml_models["predictor"] = FraudPredictor(model_dir="models")
    logger.info("FraudPredictor loaded successfully.")
    yield
    ml_models.clear()


app = FastAPI(
    title="Fraud Detection Monitoring API",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {"message": "Fraud Detection Monitoring API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    predictor = ml_models.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        data = transaction.model_dump()
        data["type"] = transaction.type.value
        result = predictor.predict(data)
        return result
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    predictor = ml_models.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not batch.transactions:
        raise HTTPException(status_code=400, detail="transactions list is empty")

    try:
        rows = []
        for txn in batch.transactions:
            row = txn.model_dump()
            row["type"] = txn.type.value
            rows.append(row)

        df = pd.DataFrame(rows)
        result_df = predictor.predict_batch(df)

        results = [
            PredictionResponse(
                is_fraud=bool(row["fraud_prediction"]),
                probability=float(row["fraud_probability"]),
                threshold=predictor.threshold,
            )
            for _, row in result_df.iterrows()
        ]

        return BatchPredictionResponse(
            results=results,
            total_transactions=len(results),
            flagged_as_fraud=sum(r.is_fraud for r in results),
        )
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {exc}")