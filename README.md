# Real-Time Financial Fraud Detection Pipeline

An end-to-end machine learning pipeline for detecting fraudulent financial transactions using the PaySim synthetic dataset. The pipeline trains, evaluates, and compares multiple classifiers with two imbalance-handling strategies, then persists the best model for production inference.

## Performance

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest (class_weight)** | **1.0000** | **0.9781** | **0.9889** | 0.9934 | 0.9845 |
| XGBoost (SMOTE) | 0.9955 | 0.9781 | 0.9867 | 0.9975 | 0.9808 |
| XGBoost (class_weight) | 0.9738 | 0.9781 | 0.9759 | 0.9979 | 0.9839 |
| Random Forest (SMOTE) | 0.9612 | 0.9781 | 0.9697 | 0.9970 | 0.9846 |
| Logistic Regression (SMOTE) | 0.8884 | 0.9781 | 0.9311 | 0.9958 | 0.9774 |
| Logistic Regression (class_weight) | 0.1056 | 0.9781 | 0.1905 | 0.9961 | 0.9544 |

**Best model:** Random Forest (class_weight) — F1 = 0.9889, Optimal threshold = 0.379

## Architecture

```
main.py                          Pipeline orchestrator (13-step workflow)
src/
  preprocessing.py               Data loading, cleaning, encoding, scaling
  feature_engineering.py          Domain-specific feature creation (9 features)
  train_model.py                  Model configuration, training, SMOTE oversampling
  evaluate.py                    Evaluation, threshold optimisation, plots, report
  predict.py                     FraudPredictor class for production inference
```

### Pipeline Stages

```
Load CSV -> Clean -> Engineer Features -> Encode -> Split -> Scale
  -> Pipeline A: Train with class_weight (LR, RF, XGBoost)
  -> Pipeline B: Train with SMOTE       (LR, RF, XGBoost)
  -> Compare all 6 models -> Select best by F1
  -> Optimise threshold -> Extract feature importance
  -> Save artifacts -> Generate plots + report
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone <repo-url>
cd Real-Time-Financial-Fraud-Detection-Pipeline
pip install -r requirements.txt
```

Place the dataset at `data/transactions_train.csv`.

## Usage

### Train the pipeline

```bash
python main.py
```

This runs the full 13-step pipeline and produces all artifacts in `models/`.

### Predict fraud on new transactions

```python
from src.predict import FraudPredictor

predictor = FraudPredictor(model_dir="models")

result = predictor.predict({
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "nameOrig": "C1305486145",
    "oldbalanceOrig": 181.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
})
# result -> {"is_fraud": True, "probability": 0.92, "threshold": 0.379}
```

### Batch prediction

```python
import pandas as pd
from src.predict import FraudPredictor

predictor = FraudPredictor(model_dir="models")
df = pd.read_csv("new_transactions.csv")
results = predictor.predict_batch(df)
# Adds fraud_probability and fraud_prediction columns
```

## Generated Artifacts

| File | Description |
|---|---|
| `models/best_model.joblib` | Trained Random Forest classifier |
| `models/label_encoder.joblib` | Fitted LabelEncoder for `type` column |
| `models/standard_scaler.joblib` | Fitted StandardScaler (16 features) |
| `models/feature_names.joblib` | Ordered list of feature column names |
| `models/optimal_threshold.joblib` | Optimised classification threshold |
| `models/confusion_matrices.png` | Confusion matrices for all 6 models |
| `models/feature_importance.png` | Feature importance bar chart |
| `models/training.log` | Full pipeline execution log |
| `training_report.md` | Auto-generated Markdown training report |

## Engineered Features

| Feature | Description |
|---|---|
| `balance_delta_orig` | Sender balance change |
| `balance_delta_dest` | Receiver balance change |
| `orig_balance_zeroed` | 1 if sender balance drained to zero |
| `balance_mismatch_orig` | 1 if origin balance change != -amount |
| `balance_mismatch_dest` | 1 if destination balance change != +amount |
| `amount_log` | log1p(amount) — reduces skew |
| `amount_ratio_orig` | amount / (oldbalanceOrig + 1) |
| `hour_of_day` | step % 24 — captures diurnal patterns |
| `is_fraud_prone_type` | 1 if type is TRANSFER or CASH_OUT |

## Tech Stack

- **Python 3.13** — Core language
- **pandas 3.0** — Data manipulation
- **scikit-learn 1.9** — ML models, preprocessing, evaluation
- **XGBoost 3.3** — Gradient boosting classifier
- **imbalanced-learn 0.14** — SMOTE oversampling
- **matplotlib / seaborn** — Visualisation
## Monitoring Setup (Week 4)

### Monitoring Stack

* **Grafana** — Used for creating real-time monitoring dashboards.
* **Prometheus** — Used for collecting and storing monitoring metrics.
* **Windows Exporter** — Used for collecting Windows system metrics such as CPU and memory usage.

### Local Access

**Grafana Dashboard:**
http://localhost:3000

**Prometheus Server:**
http://localhost:9090

### Configuration Status

* Grafana installed and configured successfully.
* Prometheus installed and running successfully.
* Prometheus connected as a Grafana data source.
* FastAPI monitoring integration added using `prometheus_fastapi_instrumentator`.
* `/` and `/health` monitoring endpoints created.
* Windows Exporter configured on port `9182` with `cpu`, `memory`, `os`, `net`, `logical_disk`, and `system` collectors enabled.
* Prometheus configured to scrape Windows Exporter metrics.
* Prometheus configured to scrape FastAPI metrics on port `8000`.
* `windows-exporter`, `prometheus`, and `fastapi` targets all verified as **UP** in Prometheus.
* Grafana dashboard built and saved with live panels for CPU Usage, Memory Usage, and Fraud API Health.
* Dashboard configuration exported and version-controlled as `dashboard.json`.
* Monitoring setup completed for real-time visualization of fraud detection and system performance metrics.

### Grafana Dashboard

**File:** `dashboard.json`
**Screenshot:** `Monitoring_Dashboard.jpeg`

**Panels included:**

| Panel | Metric Source | Query |
|---|---|---|
| CPU Usage (%) | Windows Exporter | `100 - (avg(rate(windows_cpu_time_total{mode="idle"}[5m])) * 100)` |
| Memory Usage (%) | Windows Exporter | `(1 - (windows_memory_physical_free_bytes / windows_memory_physical_total_bytes)) * 100` |
| Fraud API Health | FastAPI (`prometheus_fastapi_instrumentator`) | `up{job="fastapi"}` |

**To restore this dashboard:**
1. Open Grafana → **Dashboards** → **New** → **Import**
2. Upload `dashboard.json`
3. Select **Prometheus** as the data source when prompted
