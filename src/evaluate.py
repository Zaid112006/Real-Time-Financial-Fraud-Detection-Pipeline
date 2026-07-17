"""
Evaluation Module — Real-Time Financial Fraud Detection Pipeline
================================================================

Provides comprehensive model evaluation, threshold optimisation,
feature-importance extraction, and automated Markdown report generation.

Metrics computed for each model:
    * Precision  (fraud class)
    * Recall     (fraud class)
    * F1-score   (fraud class)
    * ROC-AUC
    * PR-AUC     (Precision–Recall AUC — more informative for imbalanced
      data than ROC-AUC)
    * Confusion matrix

The best model is selected by **F1-score on the fraud class** (label=1),
which balances the trade-off between catching fraud (recall) and
avoiding false alarms (precision).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / CI runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ── Single-Model Evaluation ─────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a single model on the test set.

    Args:
        name: Human-readable model name.
        model: Fitted estimator with ``predict`` and ``predict_proba``.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary containing all evaluation metrics and raw predictions.
    """
    y_pred = model.predict(X_test)

    # Probability estimates for AUC metrics
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    # Core metrics — fraud class = 1
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc = float(roc_auc_score(y_test, y_proba))

    # PR-AUC (more informative under extreme imbalance)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    pr = float(auc(rec_curve, prec_curve))

    cm = confusion_matrix(y_test, y_pred)

    metrics: Dict[str, Any] = {
        "name": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    logger.info(
        "  %-35s  Prec=%.4f  Rec=%.4f  F1=%.4f  ROC=%.4f  PR=%.4f",
        name, precision, recall, f1, roc, pr,
    )
    return metrics


# ── Multi-Model Comparison ───────────────────────────────────────────────────

def compare_models(
    results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a comparison table of all evaluated models.

    Args:
        results: Dict mapping *model name* → ``{"model": …, "metrics": …}``.

    Returns:
        DataFrame with one row per model, sorted by F1-score descending.
    """
    rows = []
    for name, data in results.items():
        m = data["metrics"]
        rows.append({
            "Model": name,
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        })

    comparison = (
        pd.DataFrame(rows)
        .sort_values("F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\n%s", comparison.to_string(index=False))
    return comparison


def select_best_model(
    results: Dict[str, Dict[str, Any]],
    metric: str = "f1_score",
) -> str:
    """Select the best model by a given metric.

    Args:
        results: Dict mapping model name → ``{"model": …, "metrics": …}``.
        metric: Metric key to maximise (default ``f1_score``).

    Returns:
        Name of the best-performing model.
    """
    best_name = max(results, key=lambda n: results[n]["metrics"][metric])
    best_val = results[best_name]["metrics"][metric]
    logger.info("* Best model by %s: %s (%.4f)", metric, best_name, best_val)
    return best_name


# ── Threshold Optimisation ───────────────────────────────────────────────────

def optimize_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_thresholds: int = 200,
) -> Tuple[float, Dict[str, float]]:
    """Find the classification threshold that maximises F1 for fraud.

    Sweeps thresholds from 0.01 to 0.99 and selects the one with the
    highest F1-score on the positive (fraud) class.

    Args:
        model: Fitted model with ``predict_proba``.
        X_test: Test features.
        y_test: True labels.
        n_thresholds: Number of threshold values to evaluate.

    Returns:
        ``(optimal_threshold, metrics_at_optimal_threshold)``
    """
    if not hasattr(model, "predict_proba"):
        logger.warning(
            "Model lacks predict_proba — returning default threshold 0.5"
        )
        return 0.5, {}

    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = {
                "threshold": best_threshold,
                "precision": float(
                    precision_score(y_test, y_pred, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, y_pred, zero_division=0)
                ),
                "f1_score": float(f1),
            }

    logger.info(
        "Optimal threshold: %.3f  ->  Prec=%.4f  Rec=%.4f  F1=%.4f",
        best_threshold,
        best_metrics.get("precision", 0),
        best_metrics.get("recall", 0),
        best_metrics.get("f1_score", 0),
    )
    return best_threshold, best_metrics


# ── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract feature importances from a trained model.

    Supports tree-based models (``feature_importances_``) and linear
    models (``coef_``).

    Args:
        model: Fitted estimator.
        feature_names: Ordered list of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns ``['feature', 'importance']``,
        sorted descending.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not expose feature importances")
        return pd.DataFrame(columns=["feature", "importance"])

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Top %d features:\n%s", top_n, fi.to_string(index=False))
    return fi


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    save_dir: str | Path = "models",
) -> str:
    """Plot confusion matrices for all models in a single figure.

    Args:
        results: Dict mapping model name → ``{"metrics": …}``.
        save_dir: Directory to save the plot.

    Returns:
        Path to the saved image.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False,
    )

    for idx, (name, data) in enumerate(results.items()):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        cm = data["metrics"]["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        short_name = name[:30] + "…" if len(name) > 30 else name
        ax.set_title(short_name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # Hide unused sub-plots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    save_path = save_dir / "confusion_matrices.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrices to %s", save_path)
    return str(save_path)


def plot_feature_importance(
    importances: pd.DataFrame,
    save_dir: str | Path = "models",
) -> str:
    """Plot feature importance as a horizontal bar chart.

    Args:
        importances: DataFrame from ``get_feature_importance``.
        save_dir: Directory to save the plot.

    Returns:
        Path to the saved image.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    data = importances.sort_values("importance", ascending=True)
    ax.barh(data["feature"], data["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Best Model")
    plt.tight_layout()

    save_path = save_dir / "feature_importance.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved feature importance plot to %s", save_path)
    return str(save_path)


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(
    dataset_stats: Dict[str, Any],
    comparison_df: pd.DataFrame,
    best_model_name: str,
    best_model_metrics: Dict[str, Any],
    threshold_info: Dict[str, float],
    feature_importances: pd.DataFrame,
    save_path: str | Path = "training_report.md",
) -> None:
    """Generate a comprehensive Markdown training report.

    Args:
        dataset_stats: Output of ``inspect_dataset``.
        comparison_df: Model comparison DataFrame.
        best_model_name: Name of the selected best model.
        best_model_metrics: Metrics dict for the best model.
        threshold_info: Output of ``optimize_threshold``.
        feature_importances: DataFrame from ``get_feature_importance``.
        save_path: Where to write the Markdown file.
    """
    lines: List[str] = []
    _a = lines.append  # shorthand

    _a("# Fraud Detection — Training Report\n")
    _a("*Generated automatically by the ML training pipeline*\n")

    # ── 1. Dataset ───────────────────────────────────────────────────────
    _a("## 1. Dataset Summary\n")
    _a("| Metric | Value |")
    _a("|---|---|")
    _a(f"| Rows | {dataset_stats['n_rows']:,} |")
    _a(f"| Columns | {dataset_stats['n_columns']} |")
    _a(f"| Missing Values | {dataset_stats['total_missing']} |")
    _a(f"| Fraud Rate | {dataset_stats['fraud_rate'] * 100:.4f}% |")
    _a(f"| Imbalance Ratio | 1:{dataset_stats['imbalance_ratio']} |")
    td = dataset_stats["target_distribution"]
    _a(f"| Legitimate Transactions | {td.get(0, 'N/A'):,} |")
    _a(f"| Fraudulent Transactions | {td.get(1, 'N/A'):,} |")
    _a("")

    # ── 2. Model Comparison ──────────────────────────────────────────────
    _a("## 2. Model Comparison\n")
    try:
        _a(comparison_df.to_markdown(index=False))
    except ImportError:
        _a(comparison_df.to_string(index=False))
    _a("")

    # ── 3. Best Model ────────────────────────────────────────────────────
    _a("## 3. Best Model\n")
    _a(f"**Selected Model:** {best_model_name}\n")
    _a("**Selection Criterion:** Highest F1-Score on fraud class "
       "(positive label = 1)\n")

    bm = best_model_metrics
    _a("| Metric | Value |")
    _a("|---|---|")
    _a(f"| Precision | {bm.get('precision', 0):.4f} |")
    _a(f"| Recall | {bm.get('recall', 0):.4f} |")
    _a(f"| F1-Score | {bm.get('f1_score', 0):.4f} |")
    _a(f"| ROC-AUC | {bm.get('roc_auc', 0):.4f} |")
    _a(f"| PR-AUC | {bm.get('pr_auc', 0):.4f} |")
    _a("")

    # ── 4. Threshold ─────────────────────────────────────────────────────
    _a("## 4. Threshold Optimization\n")
    if threshold_info:
        _a(f"Default threshold: **0.50**\n")
        opt_t = threshold_info.get("threshold", 0.5)
        _a(f"Optimized threshold: **{opt_t:.3f}**\n")
        _a("| Metric | Default (0.5) | Optimized |")
        _a("|---|---|---|")
        _a(f"| Precision | {bm.get('precision', 0):.4f} "
           f"| {threshold_info.get('precision', 0):.4f} |")
        _a(f"| Recall | {bm.get('recall', 0):.4f} "
           f"| {threshold_info.get('recall', 0):.4f} |")
        _a(f"| F1-Score | {bm.get('f1_score', 0):.4f} "
           f"| {threshold_info.get('f1_score', 0):.4f} |")
    else:
        _a("Threshold optimization not available for this model.\n")
    _a("")

    # ── 5. Feature Importance ────────────────────────────────────────────
    _a("## 5. Feature Importance\n")
    if not feature_importances.empty:
        try:
            _a(feature_importances.to_markdown(index=False))
        except ImportError:
            _a(feature_importances.to_string(index=False))
    else:
        _a("Feature importance not available for this model type.\n")
    _a("")

    # ── 6. Imbalance Strategy ────────────────────────────────────────────
    _a("## 6. Imbalance Handling: class_weight vs SMOTE\n")
    _a("### Trade-off Analysis\n")
    _a("| Strategy | Pros | Cons |")
    _a("|---|---|---|")
    _a("| **class_weight** | Fast, no extra memory, no synthetic data "
       "| May under-represent minority patterns |")
    _a("| **SMOTE** | Better minority representation, can improve recall "
       "| Slower, memory-intensive, risk of overfitting |")
    _a("")
    _a("> **Recommendation:** For production fraud detection at scale, "
       "`class_weight` is generally preferred for its simplicity and lower "
       "computational cost.  SMOTE is most useful for offline analysis or "
       "when the training set is smaller.\n")

    # ── 7. Artifacts ─────────────────────────────────────────────────────
    _a("## 7. Saved Artifacts\n")
    _a("| File | Description |")
    _a("|---|---|")
    _a("| `models/best_model.joblib` | Trained best model |")
    _a("| `models/label_encoder.joblib` | Fitted LabelEncoder ('type') |")
    _a("| `models/standard_scaler.joblib` | Fitted StandardScaler |")
    _a("| `models/feature_names.joblib` | Ordered feature names |")
    _a("| `models/optimal_threshold.joblib` | Optimised threshold |")
    _a("| `models/confusion_matrices.png` | Confusion matrices |")
    _a("| `models/feature_importance.png` | Feature importance chart |")
    _a("")

    # ── Write to file ────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    Path(save_path).write_text(report_text, encoding="utf-8")
    logger.info("Training report saved to %s", save_path)
