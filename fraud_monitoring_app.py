import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator

from src.predict import FraudPredictor
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)

# Load variables from .env into the process environment (FRAUD_API_KEY, etc.)
load_dotenv()

logger = logging.getLogger(__name__)

ml_models = {}

# -- API key auth setup --------------------------------------------
API_KEY = os.getenv("FRAUD_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(api_key_header)) -> str:
    """FastAPI dependency: checks the X-API-Key header against FRAUD_API_KEY.

    Raises 401 if missing/incorrect. Raises 500 if the server itself has
    no FRAUD_API_KEY configured (misconfiguration, not a client error).
    """
    if not API_KEY:
        logger.error("FRAUD_API_KEY is not set on the server (.env missing?)")
        raise HTTPException(
            status_code=500, detail="Server misconfiguration: API key not set"
        )
    if not provided_key or provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return provided_key


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
def predict(
    transaction: TransactionRequest,
    api_key: str = Depends(verify_api_key),
):
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
def predict_batch(
    batch: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key),
):
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