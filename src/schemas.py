"""
API Schemas — Real-Time Financial Fraud Detection Pipeline
============================================================
Pydantic models used by the FastAPI app to validate incoming
requests and document response shapes.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=0, description="Hour-step in the simulation")
    type: TransactionType
    amount: float = Field(..., ge=0)
    nameOrig: str
    oldbalanceOrig: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C1305486145",
                "oldbalanceOrig": 181.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    probability: float
    threshold: float


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_transactions: int
    flagged_as_fraud: int