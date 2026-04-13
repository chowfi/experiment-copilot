"""
Customer churn binary classifier using RandomForest + sklearn Pipeline.

Dependencies:
    pip install pandas scikit-learn
    Or from repo root: uv sync && uv run python customer_churn_random_forest.py

Data (Kaggle):
    https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
    The Kaggle-provided train/test split has severe distribution mismatch, so we
    concatenate both CSVs and re-split with a stratified 80/20 split ourselves.

Fetch CSVs into ./data (next to this file) with:
    bash download_customer_churn_data.sh
Requires ~/.kaggle/kaggle.json (Kaggle account API token). Or set DATA_DIR if you unpack elsewhere.

Logging: INFO by default; set LOG_LEVEL=DEBUG (or WARNING, etc.) to adjust verbosity.
Logs go to stderr and to a file (default: customer_churn_random_forest.log next to this script).
Override the file path with LOG_FILE=/path/to/run.log
Uses a dedicated logger (customer_churn_rf) so file logging is not lost if the root logger is reset during parallel sklearn/joblib work.

Binary decisions: by default we predict churn (positive class) when P(churn) >= DECISION_THRESHOLD.
Lower threshold → more predicted 1s, fewer 0s. Override with CHURN_DECISION_THRESHOLD (e.g. 0.35).

Features: `Gender`, `Subscription Type`, and `Contract Length` are one-hot encoded; all other feature columns are numeric (median impute + scale).
`CustomerID` is excluded (identifier; strong spurious correlation with churn in this split).

Grid search: set RUN_GRID_SEARCH=1 (optional: GRID_CV_SPLITS=3, GRID_SEARCH_VERBOSE=1). Heavy on full train data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Dedicated name so logs are not dropped when the root logger is reset (e.g. by parallel workers).
LOGGER_NAME = "customer_churn_rf"
logger = logging.getLogger(LOGGER_NAME)

# --- paths (edit after placing Kaggle files) ---
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FILE = "customer_churn_dataset-training-master.csv"
TEST_FILE = "customer_churn_dataset-testing-master.csv"

TARGET_COL = "Churn"

# Identifiers / columns not used as predictors (see module docstring).
EXCLUDE_FROM_FEATURES: frozenset[str] = frozenset({"CustomerID"})

# Categorical columns: always one-hot encoded (stringified before OHE).
CATEGORICAL_OHE_COLS = ["Gender", "Subscription Type", "Contract Length"]

# Grid search (expensive on full data). Enable with RUN_GRID_SEARCH=1.
PARAM_GRID: dict[str, list] = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 12, 24],
    "clf__min_samples_split": [2, 32],
    "clf__min_samples_leaf": [1, 20],
    "clf__max_features": ["sqrt", 0.25],
}

# --- five tunable Random Forest hyperparameters ---
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
}

# Fixed RandomForest args (not in RF_PARAMS); included in experiment logs.
RF_FIXED = {
    "random_state": 42,
    "n_jobs": -1,
}

# Predict positive (churn) when P(pos) >= this; lower → more 1s, fewer 0s vs 0.5.
DECISION_THRESHOLD = 0.146465


def configure_logging() -> Path:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    default_log = Path(__file__).resolve().parent / "customer_churn_random_forest.log"
    log_path = Path(os.environ.get("LOG_FILE", str(default_log))).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    app_log = logging.getLogger(LOGGER_NAME)
    app_log.setLevel(level)
    for h in list(app_log.handlers):
        app_log.removeHandler(h)
        h.close()
    app_log.propagate = False

    stream = logging.StreamHandler()
    stream.setLevel(level)
    stream.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    app_log.addHandler(stream)
    app_log.addHandler(file_handler)
    return log_path


def _predict_from_pos_proba(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int64, copy=False)


def _log_classification_metrics(split: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> None:
    prefix = f"Results [{split}]"
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    logger.info("%s: accuracy=%.6f", prefix, acc)
    try:
        auc = roc_auc_score(y_true, y_score)
        logger.info("%s: roc_auc=%.6f", prefix, auc)
    except ValueError as e:
        logger.warning("%s: roc_auc skipped (%s)", prefix, e)
    logger.info("%s: classification_report\n%s", prefix, report)


def _feature_column_names(columns: pd.Index) -> list[str]:
    return [c for c in columns if c != TARGET_COL and c not in EXCLUDE_FROM_FEATURES]


def _encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder | None]:
    if y_train.dtype == object or str(y_train.dtype) == "string":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_train.astype(str))
        y_te = le.transform(y_test.astype(str))
        logger.info("Target encoded with LabelEncoder; classes=%s", list(le.classes_))
        return y_tr, y_te, le
    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()
    logger.info("Target used as numeric/array dtype=%s", y_train.dtype)
    return y_tr, y_te, None


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat_ohe", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No feature columns after excluding target.")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = RandomForestClassifier(
        **RF_FIXED,
        **RF_PARAMS,
    )
    logger.info(
        "Preprocessor: numeric_features=%d one_hot_features=%s (%d cols)",
        len(numeric_cols),
        categorical_cols,
        len(categorical_cols),
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def main() -> None:
    log_path = configure_logging()
    try:
        logger.info("Log file: %s", log_path)
        logger.info(
            "Experiment: data_dir=%s train_file=%s test_file=%s target=%s",
            DATA_DIR,
            TRAIN_FILE,
            TEST_FILE,
            TARGET_COL,
        )
        logger.info("Hyperparameters RF_PARAMS=%s", RF_PARAMS)
        logger.info("Hyperparameters RF_FIXED=%s", RF_FIXED)
        logger.info("Full classifier kwargs=%s", {**RF_FIXED, **RF_PARAMS})
        decision_threshold = float(os.environ.get("CHURN_DECISION_THRESHOLD", str(DECISION_THRESHOLD)))
        logger.info(
            "Decision threshold: P(positive) >= %.6f → predict 1 (env CHURN_DECISION_THRESHOLD overrides DECISION_THRESHOLD)",
            decision_threshold,
        )

        train_path = DATA_DIR / TRAIN_FILE
        test_path = DATA_DIR / TEST_FILE
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"Expected train and test CSVs under {DATA_DIR}:\n"
                f"  {TRAIN_FILE}\n  {TEST_FILE}\n"
                "Set DATA_DIR to your Kaggle unzip folder."
            )

        logger.info("Loading CSVs: %s", train_path)
        df_kaggle_train = pd.read_csv(train_path)
        logger.info("Loading CSVs: %s", test_path)
        df_kaggle_test = pd.read_csv(test_path)
        logger.info(
            "Loaded raw rows: kaggle_train=%d kaggle_test=%d",
            len(df_kaggle_train),
            len(df_kaggle_test),
        )

        df_all = pd.concat([df_kaggle_train, df_kaggle_test], ignore_index=True)
        logger.info("Combined dataset: %d rows", len(df_all))

        if TARGET_COL not in df_all.columns:
            raise KeyError(
                f"Column {TARGET_COL!r} not in data. Columns: {list(df_all.columns)}"
            )

        _n_before = len(df_all)
        df_all = df_all.dropna(subset=[TARGET_COL])
        _n_dropped = _n_before - len(df_all)
        if _n_dropped:
            logger.warning("Dropped %d rows with missing %s", _n_dropped, TARGET_COL)

        test_size = float(os.environ.get("TEST_SIZE", "0.2"))
        df_train, df_test = train_test_split(
            df_all, test_size=test_size, random_state=42, stratify=df_all[TARGET_COL],
        )
        logger.info(
            "Stratified split (test_size=%.2f): train=%d test=%d",
            test_size, len(df_train), len(df_test),
        )

        feature_cols = _feature_column_names(df_train.columns)
        excluded = [c for c in df_train.columns if c != TARGET_COL and c not in feature_cols]
        if excluded:
            logger.info("Excluded from features: %s", excluded)
        X_train = df_train[feature_cols].copy()
        X_test = df_test[feature_cols].copy()
        y_train_raw = df_train[TARGET_COL]
        y_test_raw = df_test[TARGET_COL]

        y_train, y_test, _target_encoder = _encode_target(y_train_raw, y_test_raw)

        logger.info("Train shape=%s Test shape=%s", X_train.shape, X_test.shape)
        logger.info("Target (train) value counts:\n%s", y_train_raw.value_counts())

        missing_ohe = [c for c in CATEGORICAL_OHE_COLS if c not in feature_cols]
        if missing_ohe:
            raise KeyError(
                f"Expected categorical columns for one-hot encoding missing from data: {missing_ohe}. "
                f"Available: {feature_cols}"
            )
        categorical_cols = [c for c in CATEGORICAL_OHE_COLS if c in feature_cols]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        for col in categorical_cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        run_grid = os.environ.get("RUN_GRID_SEARCH", "").lower() in ("1", "true", "yes")
        grid_cv_splits = int(os.environ.get("GRID_CV_SPLITS", "3"))

        if run_grid:
            logger.info(
                "GridSearchCV enabled: cv=%d param_grid keys=%s",
                grid_cv_splits,
                list(PARAM_GRID.keys()),
            )
            cv = StratifiedKFold(
                n_splits=grid_cv_splits, shuffle=True, random_state=42
            )
            base = build_pipeline(numeric_cols, categorical_cols)
            grid = GridSearchCV(
                base,
                param_grid=PARAM_GRID,
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
                refit=True,
                verbose=int(os.environ.get("GRID_SEARCH_VERBOSE", "1")),
                return_train_score=False,
            )
            logger.info("GridSearchCV fitting (this may take a long time on full training data)")
            grid.fit(X_train, y_train)
            logger.info("GridSearchCV best_params=%s", grid.best_params_)
            logger.info("GridSearchCV best_cv_roc_auc=%.6f", grid.best_score_)
            pipeline = grid.best_estimator_
        else:
            pipeline = build_pipeline(numeric_cols, categorical_cols)
            logger.info("Fitting pipeline on training set (set RUN_GRID_SEARCH=1 for GridSearchCV)")
            pipeline.fit(X_train, y_train)
            logger.info("Fit complete")

        def _scores(X):
            proba = pipeline.predict_proba(X)
            pos_idx = 1 if proba.shape[1] > 1 else 0
            return proba[:, pos_idx]

        y_score_train = _scores(X_train)
        y_score_test = _scores(X_test)
        y_pred_train = _predict_from_pos_proba(y_score_train, decision_threshold)
        y_pred_test = _predict_from_pos_proba(y_score_test, decision_threshold)
        _log_classification_metrics("train", y_train, y_pred_train, y_score_train)
        _log_classification_metrics("test", y_test, y_pred_test, y_score_test)

    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
