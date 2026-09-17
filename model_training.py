"""
model_training.py
------------------
Trains an XGBoost classifier to predict customer churn on the preprocessed
Telco dataset, evaluates it, and persists everything the Streamlit app
(`app.py`) needs to reuse it later.

Steps performed:
    1. Preprocess the raw data via `data_preprocessing.preprocess_data`.
    2. Stratified 80/20 train/test split (random_state=42).
    3. Train an XGBoost classifier with `scale_pos_weight` set to counter
       class imbalance.
    4. Evaluate with ROC-AUC, precision, recall, F1, and a confusion matrix.
    5. Save the trained model, the label encoders, the training feature
       column order, and per-feature default values (used by the app to
       fill in inputs the user doesn't provide) to disk with joblib.
"""

import logging

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_preprocessing import DATA_URL, TARGET_COLUMN, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_PATH = "model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
DEFAULT_VALUES_PATH = "default_values.pkl"


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split so the churn rate is preserved in both sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio of negative to positive samples, used to counter class imbalance."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        logger.warning("No positive samples found in training data; defaulting scale_pos_weight to 1.0")
        return 1.0
    return negatives / positives


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """Train an XGBoost classifier tuned for an imbalanced binary target."""
    scale_pos_weight = compute_scale_pos_weight(y_train)
    logger.info("Training XGBoost with scale_pos_weight=%.3f", scale_pos_weight)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute ROC-AUC, precision, recall, F1, and the confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print the evaluation metrics to the console."""
    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"ROC-AUC Score : {metrics['roc_auc']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1 Score      : {metrics['f1']:.4f}")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(metrics["confusion_matrix"])
    print("=" * 50)


def compute_default_values(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict:
    """Compute a median/mode default for every feature.

    Used by the Streamlit app to fill in a full feature row for inputs the
    sidebar doesn't explicitly collect from the user.
    """
    defaults = {}
    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode().iloc[0]
    return defaults


def main() -> None:
    try:
        df, encoders = preprocess_data(DATA_URL)
    except Exception as exc:  # noqa: BLE001
        logger.error("Preprocessing failed: %s", exc)
        raise

    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

    defaults = compute_default_values(df)

    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(list(X_train.columns), FEATURE_COLUMNS_PATH)
        joblib.dump(defaults, DEFAULT_VALUES_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save model artifacts: %s", exc)
        raise

    logger.info("Saved trained model to '%s'", MODEL_PATH)
    logger.info("Saved label encoders to '%s'", ENCODERS_PATH)
    logger.info("Saved feature column order to '%s'", FEATURE_COLUMNS_PATH)
    logger.info("Saved default feature values to '%s'", DEFAULT_VALUES_PATH)


if __name__ == "__main__":
    main()
