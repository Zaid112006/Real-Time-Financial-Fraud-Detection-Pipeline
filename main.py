"""
Main Pipeline Orchestrator — Real-Time Financial Fraud Detection Pipeline
=========================================================================

Runs the complete ML training pipeline end-to-end:

    1.  Load and inspect the dataset
    2.  Clean data (drop identifiers, handle missing values)
    3.  Engineer domain-specific features
    4.  Encode categorical variables
    5.  Split into train / test (stratified)
    6.  Scale features
    7.  Pipeline A — train models with ``class_weight``
    8.  Pipeline B — train models with SMOTE oversampling
    9.  Compare all models across both pipelines
    10. Select the best model (by F1-score on fraud class)
    11. Optimise the classification threshold
    12. Extract feature importances
    13. Save all artifacts
    14. Generate plots and Markdown training report

Usage::

    python main.py
"""

import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    TARGET_COLUMN,
    load_dataset,
    inspect_dataset,
    clean_data,
    encode_categoricals,
    scale_features,
    save_artifacts,
)
from src.feature_engineering import engineer_features
from src.train_model import get_models, train_model, apply_smote, save_model
from src.evaluate import (
    compare_models,
    evaluate_model,
    generate_report,
    get_feature_importance,
    optimize_threshold,
    plot_confusion_matrices,
    plot_feature_importance,
    select_best_model,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = Path("data/transactions_train.csv")
MODELS_DIR = Path("models")
REPORT_PATH = Path("training_report.md")
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_STRATEGY = 0.1  # minority → 10% of majority after resampling


def _setup_logging() -> None:
    """Configure root logger with console and file handlers."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                MODELS_DIR / "training.log", mode="w", encoding="utf-8",
            ),
        ],
    )


def main() -> None:
    """Execute the full training pipeline."""
    _setup_logging()
    logger = logging.getLogger("FraudPipeline")

    try:
        pipeline_start = time.time()
        logger.info("=" * 70)
        logger.info("FRAUD DETECTION — ML TRAINING PIPELINE")
        logger.info("=" * 70)

        # ── 1. Load & Inspect ────────────────────────────────────────────────
        logger.info("\n>>> STEP 1: Loading dataset")
        df = load_dataset(DATA_PATH)
        dataset_stats = inspect_dataset(df)

        # ── 2. Clean ─────────────────────────────────────────────────────────
        logger.info("\n>>> STEP 2: Cleaning data")
        df = clean_data(df)

        # ── 3. Feature Engineering (before encoding) ─────────────────────────
        logger.info("\n>>> STEP 3: Feature engineering")
        df = engineer_features(df)

        # ── 4. Encode Categoricals ───────────────────────────────────────────
        logger.info("\n>>> STEP 4: Encoding categoricals")
        df, encoder = encode_categoricals(df)

        # ── 5. Train–Test Split ──────────────────────────────────────────────
        logger.info("\n>>> STEP 5: Train-test split (stratified)")
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        logger.info(
            "Train: %d samples (%d fraud)  |  Test: %d samples (%d fraud)",
            len(X_train), int(y_train.sum()),
            len(X_test), int(y_test.sum()),
        )

        # ── 6. Scale Features ────────────────────────────────────────────────
        logger.info("\n>>> STEP 6: Scaling features")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # ── 7A. Pipeline A — class_weight ────────────────────────────────────
        logger.info("\n>>> STEP 7A: Training models with class_weight")
        logger.info("=" * 60)

        scale_pos = int(
            (y_train == 0).sum() / max(int((y_train == 1).sum()), 1)
        )
        models_cw = get_models(use_class_weight=True, scale_pos_weight=scale_pos)

        all_results = {}
        for name, model in models_cw.items():
            label = f"{name} (class_weight)"
            trained = train_model(label, model, X_train_scaled, y_train)
            metrics = evaluate_model(label, trained, X_test_scaled, y_test)
            all_results[label] = {"model": trained, "metrics": metrics}

        # ── 7B. Pipeline B — SMOTE ───────────────────────────────────────────
        logger.info("\n>>> STEP 7B: Training models with SMOTE")
        logger.info("=" * 60)

        smote_available = True
        try:
            X_train_smote, y_train_smote = apply_smote(
                X_train_scaled, y_train,
                sampling_strategy=SMOTE_STRATEGY,
            )
        except Exception as e:
            logger.warning("SMOTE failed (%s: %s) — skipping Pipeline B", type(e).__name__, e)
            smote_available = False

        if smote_available:
            models_smote = get_models(
                use_class_weight=False, scale_pos_weight=1.0,
            )
            for name, model in models_smote.items():
                label = f"{name} (SMOTE)"
                trained = train_model(label, model, X_train_smote, y_train_smote)
                metrics = evaluate_model(label, trained, X_test_scaled, y_test)
                all_results[label] = {"model": trained, "metrics": metrics}

        # ── 8. Compare & Select Best ─────────────────────────────────────────
        logger.info("\n>>> STEP 8: Model comparison")
        logger.info("=" * 60)
        comparison_df = compare_models(all_results)
        best_name = select_best_model(all_results, metric="f1_score")
        best_model = all_results[best_name]["model"]
        best_metrics = all_results[best_name]["metrics"]

        # ── 9. Threshold Optimisation ────────────────────────────────────────
        logger.info("\n>>> STEP 9: Threshold optimisation")
        optimal_threshold, threshold_metrics = optimize_threshold(
            best_model, X_test_scaled, y_test,
        )
        joblib.dump(optimal_threshold, MODELS_DIR / "optimal_threshold.joblib")

        # ── 10. Feature Importance ───────────────────────────────────────────
        logger.info("\n>>> STEP 10: Feature importance")
        importances = get_feature_importance(best_model, feature_names)

        # ── 11. Save Artifacts ───────────────────────────────────────────────
        logger.info("\n>>> STEP 11: Saving artifacts")
        save_model(best_model, MODELS_DIR / "best_model.joblib")
        save_artifacts(encoder, scaler, feature_names, MODELS_DIR)

        # ── 12. Plots ────────────────────────────────────────────────────────
        logger.info("\n>>> STEP 12: Generating plots")
        plot_confusion_matrices(all_results, MODELS_DIR)
        if not importances.empty:
            plot_feature_importance(importances, MODELS_DIR)

        # ── 13. Generate Report ──────────────────────────────────────────────
        logger.info("\n>>> STEP 13: Generating training report")
        generate_report(
            dataset_stats=dataset_stats,
            comparison_df=comparison_df,
            best_model_name=best_name,
            best_model_metrics=best_metrics,
            threshold_info=threshold_metrics,
            feature_importances=importances,
            save_path=REPORT_PATH,
        )

        # ── Summary ──────────────────────────────────────────────────────────
        elapsed = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
        logger.info(
            "Best model : %s  (F1=%.4f)", best_name, best_metrics["f1_score"],
        )
        logger.info("Opt. threshold : %.3f", optimal_threshold)
        logger.info("Artifacts      : %s/", MODELS_DIR.resolve())
        logger.info("Report         : %s", REPORT_PATH.resolve())
        logger.info("=" * 70)

    except Exception:
        logger.exception("PIPELINE FAILED — see traceback above")
        sys.exit(1)


if __name__ == "__main__":
    main()
