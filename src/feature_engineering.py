"""
Feature Engineering Module — Real-Time Financial Fraud Detection Pipeline
=========================================================================

Creates domain-specific features that capture fraud signals in financial
transactions.  Each feature group is implemented as a separate function
for testability and reuse.

Feature Groups
--------------
Balance Features
    Capture balance changes and anomalies (deltas, zero-drain flags,
    mismatch indicators).
Amount Features
    Log-scale normalisation and amount-to-balance ratios.
Temporal Features
    Hour-of-day derived from the ``step`` column (1 step ≈ 1 hour).
Type Features
    Binary flag for fraud-prone transaction types (TRANSFER, CASH_OUT).

Design Decisions
----------------
* Features are computed from raw numeric columns **before scaling** so
  that domain relationships are preserved.
* ``is_fraud_prone_type`` encodes the strong prior that fraud only
  appears in TRANSFER and CASH_OUT types.  It is derived **before**
  the ``type`` column is label-encoded.
* All functions accept and return DataFrames, making them composable.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Balance Features ─────────────────────────────────────────────────────────

def create_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from sender / receiver balance changes.

    New columns created:
        ``balance_delta_orig``
            newbalanceOrig − oldbalanceOrig
        ``balance_delta_dest``
            newbalanceDest − oldbalanceDest
        ``orig_balance_zeroed``
            1 if sender's balance was drained to zero (fraud signal)
        ``balance_mismatch_orig``
            1 if origin balance change ≠ −amount (discrepancy flag)
        ``balance_mismatch_dest``
            1 if destination balance change ≠ +amount (discrepancy flag)

    Args:
        df: DataFrame containing balance columns.

    Returns:
        DataFrame with new balance features appended.
    """
    df = df.copy()

    # Balance deltas
    if {"oldbalanceOrig", "newbalanceOrig"}.issubset(df.columns):
        df["balance_delta_orig"] = (
            df["newbalanceOrig"] - df["oldbalanceOrig"]
        )

    if {"oldbalanceDest", "newbalanceDest"}.issubset(df.columns):
        df["balance_delta_dest"] = (
            df["newbalanceDest"] - df["oldbalanceDest"]
        )

    # Account draining — a strong fraud indicator
    if {"oldbalanceOrig", "newbalanceOrig"}.issubset(df.columns):
        df["orig_balance_zeroed"] = (
            (df["oldbalanceOrig"] > 0) & (df["newbalanceOrig"] == 0)
        ).astype(np.int8)

    # Balance mismatch flags — detect when the arithmetic doesn't add up
    if "amount" in df.columns:
        if {"oldbalanceOrig", "newbalanceOrig"}.issubset(df.columns):
            expected_new_orig = df["oldbalanceOrig"] - df["amount"]
            df["balance_mismatch_orig"] = (
                np.abs(expected_new_orig - df["newbalanceOrig"]) > 1.0
            ).astype(np.int8)

        if {"oldbalanceDest", "newbalanceDest"}.issubset(df.columns):
            expected_new_dest = df["oldbalanceDest"] + df["amount"]
            df["balance_mismatch_dest"] = (
                np.abs(expected_new_dest - df["newbalanceDest"]) > 1.0
            ).astype(np.int8)

    new_cols = [
        "balance_delta_orig", "balance_delta_dest",
        "orig_balance_zeroed",
        "balance_mismatch_orig", "balance_mismatch_dest",
    ]
    n_created = sum(1 for c in new_cols if c in df.columns)
    logger.info("Created %d balance features", n_created)
    return df


# ── Amount Features ──────────────────────────────────────────────────────────

def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive features from the transaction amount.

    New columns:
        ``amount_log``
            log1p(amount) — reduces right-skew.
        ``amount_ratio_orig``
            amount / (oldbalanceOrig + 1) — ratios near 1.0 often signal
            account draining.

    Args:
        df: DataFrame containing ``amount`` and optionally ``oldbalanceOrig``.

    Returns:
        DataFrame with amount features appended.
    """
    df = df.copy()

    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"])

        if "oldbalanceOrig" in df.columns:
            df["amount_ratio_orig"] = (
                df["amount"] / (df["oldbalanceOrig"] + 1)
            )

    logger.info("Created amount features: amount_log, amount_ratio_orig")
    return df


# ── Temporal Features ────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-of-day features from the ``step`` column.

    In the PaySim dataset each step represents one hour of simulation.
    ``hour_of_day`` maps this to a 0–23 range to capture diurnal patterns.

    Args:
        df: DataFrame containing the ``step`` column.

    Returns:
        DataFrame with temporal features appended.
    """
    df = df.copy()

    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        logger.info("Created temporal feature: hour_of_day")

    return df


# ── Type Features ────────────────────────────────────────────────────────────

def create_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag fraud-prone transaction types.

    In the training data fraud occurs **exclusively** in TRANSFER and
    CASH_OUT transactions.  This binary flag encodes that strong prior.

    .. warning::
        Must be called **before** label-encoding the ``type`` column,
        because it checks string values.

    Args:
        df: DataFrame with the raw (string) ``type`` column.

    Returns:
        DataFrame with ``is_fraud_prone_type`` appended.
    """
    df = df.copy()

    if "type" in df.columns:
        fraud_types = {"TRANSFER", "CASH_OUT"}

        # Only create from raw string values
        if not pd.api.types.is_numeric_dtype(df["type"]):
            df["is_fraud_prone_type"] = (
                df["type"].isin(fraud_types).astype(np.int8)
            )
            logger.info(
                "Created type feature: is_fraud_prone_type "
                "(from string 'type' column)"
            )
        else:
            logger.warning(
                "'type' column is already encoded (dtype=%s); "
                "skipping is_fraud_prone_type creation",
                df["type"].dtype,
            )

    return df


# ── Orchestrator ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order.

    Orchestrates the individual feature-creation functions.  Must be
    called **before** categorical encoding so that the ``type`` column
    is still in its raw string form.

    Args:
        df: Cleaned DataFrame (identifiers removed, missing values
            handled).

    Returns:
        DataFrame with all engineered features appended.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    initial_cols = set(df.columns)

    df = create_balance_features(df)
    df = create_amount_features(df)
    df = create_temporal_features(df)
    df = create_type_features(df)

    new_cols = sorted(set(df.columns) - initial_cols)
    logger.info(
        "Feature engineering complete: %d new features → %s",
        len(new_cols), new_cols,
    )

    return df
