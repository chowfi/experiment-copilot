"""
Tune P(churn) decision threshold on a stratified holdout from the training CSV,
then refit on full training data and report metrics on the held-out test CSV.

Uses the same preprocessing and `RF_PARAMS` / `RF_FIXED` as `customer_churn_random_forest.py`.

Run from repo root:
    uv run python tune_churn_threshold.py

Env (optional):
    DATA_DIR, LOG_LEVEL, LOG_FILE — same as the main churn script
    THRESHOLD_VAL_SIZE — validation fraction (default 0.2)
    THRESHOLD_TUNE_RANDOM_STATE — split seed (default 42)
    THRESHOLD_TUNE_METRIC — f1 | balanced_accuracy | accuracy (default f1)
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import customer_churn_random_forest as churn

logger = logging.getLogger(churn.LOGGER_NAME)


def _pos_proba_positive(pipeline, X) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    pos_idx = 1 if proba.shape[1] > 1 else 0
    return proba[:, pos_idx]


def _metric_fn(name: str):
    name = name.lower().strip()
    if name == "f1":

        def score(y_true, y_pred):
            return f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        return score
    if name == "balanced_accuracy":

        def score(y_true, y_pred):
            return balanced_accuracy_score(y_true, y_pred)

        return score
    if name == "accuracy":

        def score(y_true, y_pred):
            return accuracy_score(y_true, y_pred)

        return score
    raise ValueError(f"Unknown THRESHOLD_TUNE_METRIC={name!r}; use f1, balanced_accuracy, or accuracy")


def tune_threshold(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    metric_name: str,
    *,
    n_grid: int = 199,
) -> tuple[float, float]:
    """Return (best_threshold, best_metric_on_val)."""
    metric = _metric_fn(metric_name)
    lo, hi = float(np.min(val_scores)), float(np.max(val_scores))
    # Uniform grid in score space + endpoints (clipped) to catch plateaus at extremes
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, n_grid),
                np.linspace(lo, hi, min(n_grid, 50)),
            ]
        )
    )
    best_t, best_m = 0.5, -1.0
    for t in candidates:
        y_pred = churn._predict_from_pos_proba(val_scores, float(t))
        m = float(metric(y_val, y_pred))
        if m > best_m or (m == best_m and t < best_t):
            best_m, best_t = m, float(t)
    return best_t, best_m


def main() -> None:
    log_path = churn.configure_logging()
    metric_name = os.environ.get("THRESHOLD_TUNE_METRIC", "f1")
    val_size = float(os.environ.get("THRESHOLD_VAL_SIZE", "0.2"))
    rs = int(os.environ.get("THRESHOLD_TUNE_RANDOM_STATE", "42"))

    try:
        logger.info("Log file: %s", log_path)
        logger.info("Threshold tune: metric=%s val_size=%s random_state=%s", metric_name, val_size, rs)

        train_path = churn.DATA_DIR / churn.TRAIN_FILE
        test_path = churn.DATA_DIR / churn.TEST_FILE
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"Expected train and test CSVs under {churn.DATA_DIR}:\n"
                f"  {churn.TRAIN_FILE}\n  {churn.TEST_FILE}\n"
                "Set DATA_DIR if needed."
            )

        df_all = pd.concat(
            [pd.read_csv(train_path), pd.read_csv(test_path)],
            ignore_index=True,
        ).dropna(subset=[churn.TARGET_COL])

        test_size = float(os.environ.get("TEST_SIZE", "0.2"))
        df_train, df_test = train_test_split(
            df_all, test_size=test_size, random_state=42, stratify=df_all[churn.TARGET_COL],
        )

        feature_cols = churn._feature_column_names(df_train.columns)
        X_train = df_train[feature_cols]
        X_test = df_test[feature_cols]
        y_train, y_test, _ = churn._encode_target(
            df_train[churn.TARGET_COL],
            df_test[churn.TARGET_COL],
        )

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=rs,
        )
        logger.info("Split train: fit=%d val=%d", len(X_fit), len(X_val))

        pipeline = churn.build_pipeline(numeric_cols, categorical_cols)
        logger.info("Fitting pipeline on fit split for threshold search")
        pipeline.fit(X_fit, y_fit)

        val_scores = _pos_proba_positive(pipeline, X_val)
        best_t, best_val_metric = tune_threshold(val_scores, y_val, metric_name)
        logger.info(
            "Best threshold (on val, metric=%s): t=%.6f val_%s=%.6f",
            metric_name,
            best_t,
            metric_name,
            best_val_metric,
        )
        print(
            f"Best threshold (val, {metric_name}): {best_t:.6f}  "
            f"val_{metric_name}={best_val_metric:.6f}"
        )

        logger.info("Refitting pipeline on full training set")
        pipeline.fit(X_train, y_train)

        y_score_test = _pos_proba_positive(pipeline, X_test)
        baseline_default = churn._predict_from_pos_proba(y_score_test, 0.5)
        baseline_script = churn._predict_from_pos_proba(
            y_score_test,
            float(os.environ.get("CHURN_DECISION_THRESHOLD", str(churn.DECISION_THRESHOLD))),
        )
        y_pred_tuned = churn._predict_from_pos_proba(y_score_test, best_t)

        m = _metric_fn(metric_name)
        print(f"Test {metric_name} @ threshold 0.5:        {m(y_test, baseline_default):.6f}")
        print(
            f"Test {metric_name} @ script default/env:   {m(y_test, baseline_script):.6f}"
        )
        print(f"Test {metric_name} @ tuned threshold:     {m(y_test, y_pred_tuned):.6f}")

        logger.info(
            "Test %s: threshold=0.5 -> %.6f; script/env default -> %.6f; tuned=%.6f -> %.6f",
            metric_name,
            m(y_test, baseline_default),
            m(y_test, baseline_script),
            best_t,
            m(y_test, y_pred_tuned),
        )
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
