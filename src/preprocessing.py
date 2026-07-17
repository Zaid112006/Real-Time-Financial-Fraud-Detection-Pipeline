"""
Preprocessing Module — Real-Time Financial Fraud Detection Pipeline
===================================================================

Provides reusable functions for data loading, cleaning, encoding, and
scaling.  All fitted preprocessing objects (encoder, scaler) can be
serialised for consistent inference-time preprocessing.

Design Decisions
----------------
* **LabelEncoder for 'type'**: 5 distinct categories; tree-based models
  handle ordinal encoding natively and LogReg receives additional
  engineered features that compensate.
* **StandardScaler**: Applied uniformly so LogReg converges reliably.
  Tree-based models are invariant to monotonic transformations.
* **Identifier columns** (nameOrig, nameDest) are dropped: near-unique
  strings that cannot generalise and would cause overfitting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_COLUMN: str = "isFraud"
IDENTIFIER_COLUMNS: List[str] = ["nameOrig", "nameDest"]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the transaction dataset from a CSV file.

    Validates that the file exists and contains the required target column.
    Logs basic dataset statistics upon successful load.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the target column ``isFraud`` is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available: {list(df.columns)}"
        )
    return df


# ── Dataset Inspection ───────────────────────────────────────────────────────

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a statistical summary of the dataset.

    Args:
        df: Input DataFrame (typically raw, before preprocessing).

    Returns:
        Dictionary containing row count, column count, data types,
        missing-value counts, target distribution, and fraud rate.
    """
    fraud_counts = df[TARGET_COLUMN].value_counts()
    stats: Dict[str, Any] = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "target_distribution": fraud_counts.to_dict(),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "imbalance_ratio": int(
            fraud_counts.get(0, 0) / max(fraud_counts.get(1, 1), 1)
        ),
    }

    logger.info(
        "Dataset: %d rows, %d cols | Fraud rate: %.4f%% | Imbalance: 1:%d",
        stats["n_rows"],
        stats["n_columns"],
        stats["fraud_rate"] * 100,
        stats["imbalance_ratio"],
    )
    return stats


# ── Data Cleaning ────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and handle missing values.

    Steps:
        1. Drop high-cardinality identifier columns that cannot generalise.
        2. Impute numeric missing values with the column median.
        3. Impute categorical missing values with the column mode.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned copy of the DataFrame.
    """
    df = df.copy()

    # ── Drop identifier columns ──────────────────────────────────────────
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info("Dropped identifier columns: %s", cols_to_drop)

    # ── Handle missing values ────────────────────────────────────────────
    total_missing = int(df.isnull().sum().sum())

    if total_missing > 0:
        # Numeric columns (exclude target)
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET_COLUMN
        ]
        for col in numeric_cols:
            n_miss = int(df[col].isnull().sum())
            if n_miss > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(
                    "Imputed %d missing in '%s' with median %.2f",
                    n_miss, col, median_val,
                )

        # Categorical columns
        cat_cols = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns
        for col in cat_cols:
            n_miss = int(df[col].isnull().sum())
            if n_miss > 0:
                mode_val = df[col].mode().iloc[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(
                    "Imputed %d missing in '%s' with mode '%s'",
                    n_miss, col, mode_val,
                )

        logger.info("Resolved %d total missing values", total_missing)
    else:
        logger.info("No missing values detected — dataset is clean")

    return df


# ── Categorical Encoding ────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode categorical columns.

    Currently encodes the ``type`` column.  If the DataFrame contains no
    categorical columns the original data is returned unchanged.

    Args:
        df: DataFrame (must still contain the raw categorical column).
        encoder: A previously fitted ``LabelEncoder`` for inference.
                 If ``None``, a new encoder is fitted.

    Returns:
        ``(encoded DataFrame, fitted LabelEncoder)``
    """
    df = df.copy()

    cat_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if not cat_cols:
        logger.info("No categorical columns to encode")
        return df, encoder if encoder is not None else LabelEncoder()

    if "type" in cat_cols:
        if encoder is None:
            encoder = LabelEncoder()
            df["type"] = encoder.fit_transform(df["type"])
            logger.info(
                "Fitted LabelEncoder on 'type' → classes: %s",
                list(encoder.classes_),
            )
        else:
            df["type"] = encoder.transform(df["type"])
            logger.info("Applied pre-fitted LabelEncoder on 'type'")
    else:
        logger.warning(
            "Expected 'type' column not found; encoding all categoricals"
        )
        if encoder is None:
            encoder = LabelEncoder()
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
            logger.info("Encoded column '%s'", col)

    return df, encoder


# ── Feature Scaling ──────────────────────────────────────────────────────────

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardise features using z-score scaling.

    The scaler is fitted **only on the training set** to prevent data
    leakage.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix (same columns as ``X_train``).
        scaler: Pre-fitted scaler for inference.  Fits a new one when
                ``None``.

    Returns:
        ``(scaled X_train, scaled X_test, fitted StandardScaler)``
    """
    feature_cols = X_train.columns.tolist()

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        logger.info("Fitted StandardScaler on %d features", len(feature_cols))
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        logger.info("Applied pre-fitted StandardScaler")

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


# ── Artifact Persistence ────────────────────────────────────────────────────

def save_artifacts(
    encoder: LabelEncoder,
    scaler: StandardScaler,
    feature_names: List[str],
    save_dir: str | Path = "models",
) -> None:
    """Persist preprocessing objects for inference reuse.

    Args:
        encoder: Fitted ``LabelEncoder``.
        scaler: Fitted ``StandardScaler``.
        feature_names: Ordered list of feature column names.
        save_dir: Target directory (created if it does not exist).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, save_dir / "label_encoder.joblib")
    joblib.dump(scaler, save_dir / "standard_scaler.joblib")
    joblib.dump(feature_names, save_dir / "feature_names.joblib")
    logger.info("Saved preprocessing artifacts to %s/", save_dir)


def load_artifacts(
    load_dir: str | Path = "models",
) -> Tuple[LabelEncoder, StandardScaler, List[str]]:
    """Load previously saved preprocessing objects.

    Args:
        load_dir: Directory containing the saved ``.joblib`` files.

    Returns:
        ``(encoder, scaler, feature_names)``
    """
    load_dir = Path(load_dir)

    encoder: LabelEncoder = joblib.load(load_dir / "label_encoder.joblib")
    scaler: StandardScaler = joblib.load(load_dir / "standard_scaler.joblib")
    feature_names: List[str] = joblib.load(load_dir / "feature_names.joblib")

    logger.info("Loaded preprocessing artifacts from %s/", load_dir)
    return encoder, scaler, feature_names
