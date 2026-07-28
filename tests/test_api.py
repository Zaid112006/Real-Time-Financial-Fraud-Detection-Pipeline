"""
Tests for the Fraud Detection API.

Uses FastAPI's TestClient, which runs the app in-process (no need to
start uvicorn separately). Requires models/*.joblib to already exist
(run `python main.py` first) since the app loads them at startup.

Run with:
    pytest tests/ -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# Make sure the API key exists before importing the app.
os.environ["FRAUD_API_KEY"] = "test-key-12345"

from fraud_monitoring_app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """
    Create the TestClient as a context manager so FastAPI lifespan
    events run and the ML model loads correctly.
    """
    with TestClient(app) as client:
        yield client


VALID_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181,
    "nameOrig": "C1305486145",
    "oldbalanceOrig": 181,
    "newbalanceOrig": 0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0,
    "newbalanceDest": 0,
}

HEADERS = {
    "X-API-Key": "test-key-12345"
}


# -------------------------------------------------------------------
# Basic Endpoints
# -------------------------------------------------------------------

def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_returns_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# -------------------------------------------------------------------
# Authentication
# -------------------------------------------------------------------

def test_predict_without_api_key_is_rejected(client):
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 401


def test_predict_with_wrong_api_key_is_rejected(client):
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_with_correct_api_key_succeeds(client):
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
        headers=HEADERS,
    )

    assert response.status_code == 200

    body = response.json()

    assert "is_fraud" in body
    assert "probability" in body
    assert "threshold" in body


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def test_predict_rejects_zero_amount(client):
    bad_transaction = {
        **VALID_TRANSACTION,
        "amount": 0,
    }

    response = client.post(
        "/predict",
        json=bad_transaction,
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_predict_rejects_self_transfer(client):
    bad_transaction = {
        **VALID_TRANSACTION,
        "nameOrig": "C123",
        "nameDest": "C123",
    }

    response = client.post(
        "/predict",
        json=bad_transaction,
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_type(client):
    bad_transaction = {
        **VALID_TRANSACTION,
        "type": "NOT_A_REAL_TYPE",
    }

    response = client.post(
        "/predict",
        json=bad_transaction,
        headers=HEADERS,
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# Known Fraud Case
# -------------------------------------------------------------------

def test_predict_flags_known_fraud_transaction(client):
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
        headers=HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["is_fraud"] is True


# -------------------------------------------------------------------
# Batch Endpoint
# -------------------------------------------------------------------

def test_predict_batch_with_correct_key_succeeds(client):
    payload = {
        "transactions": [VALID_TRANSACTION]
    }

    response = client.post(
        "/predict/batch",
        json=payload,
        headers=HEADERS,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_transactions"] == 1


def test_predict_batch_rejects_empty_list(client):
    payload = {
        "transactions": []
    }

    response = client.post(
        "/predict/batch",
        json=payload,
        headers=HEADERS,
    )

    assert response.status_code == 400