"""
API Schemas — Real-Time Financial Fraud Detection Pipeline
============================================================
Pydantic models used by the FastAPI app to validate incoming
requests and document response shapes.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, model_validator


class TransactionType(str, Enum):
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=0, description="Hour-step in the simulation")
    type: TransactionType
    amount: float = Field(..., gt=0, description="Must be greater than 0")
    nameOrig: str
    oldbalanceOrig: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    nameDest: str
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_business_rules(self):
        """Cross-field checks that a single field's own type/range
        can't catch on its own."""
        if self.nameOrig == self.nameDest:
            raise ValueError("nameOrig and nameDest cannot be the same account")

        # Sanity cap — PaySim amounts don't realistically exceed this;
        # guards against garbage/typo input (e.g. an extra zero or two).
        MAX_REASONABLE_AMOUNT = 100_000_000
        if self.amount > MAX_REASONABLE_AMOUNT:
            raise ValueError(
                f"amount exceeds maximum allowed value of {MAX_REASONABLE_AMOUNT}"
            )

        return self

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