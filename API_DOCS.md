# API Documentation — Fraud Detection Service

Base URL (local): `http://127.0.0.1:8000`

## Authentication

All endpoints under `/predict` require an API key sent in the `X-API-Key`
header. Health, root, and metrics endpoints do not require authentication.

```
X-API-Key: <your-key-from-.env>

```

Missing or incorrect key → `401 Unauthorized`

---

## Endpoints

### `GET /`

Returns a basic status message confirming the API is running.

**Auth required:** No

**Response `200`:**
```json
{ "message": "Fraud Detection Monitoring API Running" }
```

---

### `GET /health`

Health check endpoint, used for uptime monitoring.

**Auth required:** No

**Response `200`:**
```json
{ "status": "healthy" }
```

---

### `GET /metrics`

Prometheus-formatted metrics (auto-exposed by
`prometheus_fastapi_instrumentator`). Not intended for direct human use —
scraped by Prometheus on a schedule.

**Auth required:** No

---

### `POST /predict`

Predicts whether a single transaction is fraudulent.

**Auth required:** Yes (`X-API-Key` header)

**Request body:**

| Field | Type | Constraints |
|---|---|---|
| `step` | integer | >= 0 |
| `type` | string | one of: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER` |
| `amount` | float | > 0, <= 100,000,000 |
| `nameOrig` | string | must differ from `nameDest` |
| `oldbalanceOrig` | float | >= 0 |
| `newbalanceOrig` | float | >= 0 |
| `nameDest` | string | must differ from `nameOrig` |
| `oldbalanceDest` | float | >= 0 |
| `newbalanceDest` | float | >= 0 |

**Example request:**
```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181,
  "nameOrig": "C1305486145",
  "oldbalanceOrig": 181,
  "newbalanceOrig": 0,
  "nameDest": "C553264065",
  "oldbalanceDest": 0,
  "newbalanceDest": 0
}
```

**Response `200`:**
```json
{
  "is_fraud": true,
  "probability": 1.0,
  "threshold": 0.379
}
```

**Error responses:**

| Status | Cause |
|---|---|
| `401` | Missing or invalid `X-API-Key` |
| `422` | Validation failed (bad type, `amount <= 0`, `nameOrig == nameDest`, etc.) |
| `503` | Model not yet loaded (server just started) |

---

### `POST /predict/batch`

Predicts fraud for multiple transactions in one request.

**Auth required:** Yes (`X-API-Key` header)

**Request body:**
```json
{
  "transactions": [
    { "step": 1, "type": "TRANSFER", "amount": 181, "nameOrig": "C1305486145", "oldbalanceOrig": 181, "newbalanceOrig": 0, "nameDest": "C553264065", "oldbalanceDest": 0, "newbalanceDest": 0 },
    { "step": 1, "type": "PAYMENT", "amount": 9839.64, "nameOrig": "C1231006815", "oldbalanceOrig": 170136, "newbalanceOrig": 160296.36, "nameDest": "M1979787155", "oldbalanceDest": 0, "newbalanceDest": 0 }
  ]
}
```

**Response `200`:**
```json
{
  "results": [
    { "is_fraud": true, "probability": 1.0, "threshold": 0.379 },
    { "is_fraud": false, "probability": 0.0, "threshold": 0.379 }
  ],
  "total_transactions": 2,
  "flagged_as_fraud": 1
}
```

**Error responses:** same as `/predict`, plus `400` if `transactions` is empty.

---

## Interactive documentation

FastAPI auto-generates a live, testable version of these docs at:
http://127.0.0.1:8000/docs

Use the **Authorize** button there to enter your API key once, then test any
endpoint directly from the browser.