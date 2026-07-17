"""
Model Training Module — Real-Time Financial Fraud Detection Pipeline
====================================================================

Provides model configuration, training, and SMOTE-based oversampling.

Two imbalance-handling strategies are supported:

1. **class_weight = 'balanced'** (or ``scale_pos_weight`` for XGBoost):
   Adjusts the loss function to penalise minority-class errors more
   heavily.  No additional data is created — training stays fast.

2. **SMOTE** (Synthetic Minority Oversampling Technique):
   Generates synthetic minority-class samples via k-NN interpolation.
   Uses ``sampling_strategy=0.1`` by default (minority → 10 % of
   majority) to balance memory consumption against classification gain.

Trade-offs
----------
* **class_weight** is faster, uses no extra memory, and avoids generating
  synthetic data that may not represent real-world fraud patterns.
* **SMOTE** produces a more balanced training set which can improve
  recall, but risks overfitting to synthetic neighbours and is slower
  on large datasets.
* For production fraud detection at scale, **class_weight is generally
  preferred**.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)


# ── Model Configuration ─────────────────────────────────────────────────────

def get_models(
    use_class_weight: bool = True,
    scale_pos_weight: float = 1.0,
) -> Dict[str, Any]:
    """Return a dictionary of configured model instances.

    Logistic Regression and Random Forest are always included.
    XGBoost is added when the package is available.

    Args:
        use_class_weight: When ``True``, models use built-in
            class-weighting / ``scale_pos_weight``.
        scale_pos_weight: Ratio of negative to positive samples.
            Used only by XGBoost when ``use_class_weight`` is True.

    Returns:
        Ordered dict mapping ``{model_name: model_instance}``.
    """
    models: Dict[str, Any] = {}

    # ── Logistic Regression ──────────────────────────────────────────────
    lr_params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }
    if use_class_weight:
        lr_params["class_weight"] = "balanced"

    models["Logistic Regression"] = LogisticRegression(**lr_params)

    # ── Random Forest ────────────────────────────────────────────────────
    rf_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if use_class_weight:
        rf_params["class_weight"] = "balanced"

    models["Random Forest"] = RandomForestClassifier(**rf_params)

    # ── XGBoost (optional dependency) ────────────────────────────────────
    try:
        from xgboost import XGBClassifier

        xgb_params: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        if use_class_weight:
            xgb_params["scale_pos_weight"] = scale_pos_weight

        models["XGBoost"] = XGBClassifier(**xgb_params)
        logger.info("XGBoost available and configured")
    except ImportError:
        logger.warning("XGBoost not installed — skipping")

    logger.info(
        "Configured %d models (class_weight=%s): %s",
        len(models), use_class_weight, list(models.keys()),
    )
    return models


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Fit a single model and log training duration.

    Args:
        name: Human-readable model name (for logging).
        model: Scikit-learn–compatible estimator.
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        The fitted model instance.
    """
    logger.info("Training %s on %d samples …", name, len(X_train))
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("  [OK] %s trained in %.1f s", name, elapsed)
    return model


# ── SMOTE Oversampling ───────────────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to the training set.

    Uses a conservative ``sampling_strategy`` to avoid excessive memory
    use on million-row datasets.  A value of 0.1 means the minority
    class will be ~10 % the size of the majority class after resampling.

    Args:
        X_train: Training features.
        y_train: Training labels.
        sampling_strategy: Target minority / majority ratio after
            resampling (default 0.1).
        random_state: Reproducibility seed.

    Returns:
        ``(X_resampled, y_resampled)`` with synthetic minority samples.

    Raises:
        ImportError: If ``imbalanced-learn`` is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error(
            "imbalanced-learn is required for SMOTE.  "
            "Install with:  pip install imbalanced-learn"
        )
        raise

    logger.info(
        "Applying SMOTE (strategy=%.2f) to %d samples …",
        sampling_strategy, len(X_train),
    )

    before_dist = y_train.value_counts().to_dict()
    logger.info("  Before SMOTE: %s", before_dist)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    start = time.time()
    X_res, y_res = smote.fit_resample(X_train, y_train)
    elapsed = time.time() - start

    # Restore DataFrame / Series types and column names
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    after_dist = y_res.value_counts().to_dict()
    logger.info("  After  SMOTE: %s  (%.1f s)", after_dist, elapsed)

    return X_res, y_res


# ── Persistence ──────────────────────────────────────────────────────────────

def save_model(model: Any, path: str | Path) -> None:
    """Persist a trained model to disk.

    Args:
        model: Fitted model to save.
        path: Target filepath (should end in ``.joblib``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> Any:
    """Load a previously saved model.

    Args:
        path: Path to the ``.joblib`` file.

    Returns:
        The deserialised model instance.
    """
    model = joblib.load(Path(path))
    logger.info("Loaded model from %s", path)
    return model
