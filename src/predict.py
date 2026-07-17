"""
Prediction Module — Real-Time Financial Fraud Detection Pipeline
================================================================

Provides a production-ready ``FraudPredictor`` class designed for easy
integration into Flask / FastAPI endpoints.

Usage::

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
    # result → {"is_fraud": True, "probability": 0.92, "threshold": 0.35}
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.feature_engineering import engineer_features
from src.preprocessing import IDENTIFIER_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production-ready fraud prediction interface.

    Loads all saved artifacts (model, encoder, scaler, feature names,
    threshold) and applies the **same** preprocessing pipeline used
    during training — guaranteeing consistency between training and
    inference.

    Attributes:
        model: The trained classifier.
        encoder: Fitted LabelEncoder for the ``type`` column.
        scaler: Fitted StandardScaler.
        feature_names: Ordered list of expected feature columns.
        threshold: Classification threshold (optimised for F1).
    """

    def __init__(self, model_dir: str | Path = "models") -> None:
        """Load all artifacts from the model directory.

        Args:
            model_dir: Path to directory containing saved ``.joblib``
                files.

        Raises:
            FileNotFoundError: If any required artifact is missing.
        """
        model_dir = Path(model_dir)
        logger.info("Loading prediction artifacts from %s/", model_dir)

        self.model = joblib.load(model_dir / "best_model.joblib")
        self.encoder: LabelEncoder = joblib.load(
            model_dir / "label_encoder.joblib"
        )
        self.scaler: StandardScaler = joblib.load(
            model_dir / "standard_scaler.joblib"
        )
        self.feature_names: List[str] = joblib.load(
            model_dir / "feature_names.joblib"
        )

        threshold_path = model_dir / "optimal_threshold.joblib"
        self.threshold: float = (
            joblib.load(threshold_path) if threshold_path.exists() else 0.5
        )

        logger.info(
            "FraudPredictor ready — %d features, threshold=%.3f",
            len(self.feature_names), self.threshold,
        )

    # ── Internal Preprocessing ───────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full inference preprocessing pipeline.

        Steps:
            1. Drop identifier columns.
            2. Engineer features (balance, amount, temporal, type).
            3. Label-encode ``type`` (using saved encoder).
            4. Scale features (using saved scaler).
            5. Align columns to training feature order.

        Args:
            df: Raw transaction DataFrame.

        Returns:
            Preprocessed feature matrix ready for prediction.
        """
        df = df.copy()

        # Drop identifiers
        cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Drop target if present (e.g. batch mode on labelled data)
        if TARGET_COLUMN in df.columns:
            df.drop(columns=[TARGET_COLUMN], inplace=True)

        # Feature engineering (needs raw 'type' column)
        df = engineer_features(df)

        # Encode categoricals
        if "type" in df.columns:
            df["type"] = self.encoder.transform(df["type"])

        # Ensure correct column order; fill missing with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                logger.warning("Missing feature '%s' — filled with 0", col)

        df = df[self.feature_names]

        # Scale
        df = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_names,
            index=df.index,
        )

        return df

    # ── Public API ───────────────────────────────────────────────────────

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Predict fraud for a single transaction or small batch.

        Args:
            data: A dict representing one transaction, or a DataFrame
                with one or more rows.

        Returns:
            Dictionary with keys ``is_fraud``, ``probability``,
            ``threshold``.  For single transactions the values are
            scalars; for batches they are lists.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        X = self._preprocess(df)
        proba = self.model.predict_proba(X)[:, 1]
        is_fraud = (proba >= self.threshold).astype(bool)

        if len(df) == 1:
            return {
                "is_fraud": bool(is_fraud.iloc[0] if hasattr(is_fraud, 'iloc') else is_fraud[0]),
                "probability": float(proba.iloc[0] if hasattr(proba, 'iloc') else proba[0]),
                "threshold": self.threshold,
            }

        return {
            "is_fraud": is_fraud.tolist(),
            "probability": proba.tolist(),
            "threshold": self.threshold,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud for a batch of transactions.

        Args:
            df: DataFrame of raw transactions.

        Returns:
            Copy of the input with ``fraud_probability`` and
            ``fraud_prediction`` columns appended.
        """
        result = df.copy()
        X = self._preprocess(df)

        proba = self.model.predict_proba(X)[:, 1]
        result["fraud_probability"] = proba
        result["fraud_prediction"] = (proba >= self.threshold).astype(int)

        n_flagged = int(result["fraud_prediction"].sum())
        logger.info(
            "Batch prediction: %d transactions, %d flagged as fraud (%.2f%%)",
            len(result), n_flagged, n_flagged / len(result) * 100,
        )

        return result
