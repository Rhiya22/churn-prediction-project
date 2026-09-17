"""
data_preprocessing.py
----------------------
Loads the IBM Telco Customer Churn dataset and transforms it into a clean,
model-ready dataframe.

Steps performed:
    1. Load the raw CSV (from a local path or URL).
    2. Convert `TotalCharges` to numeric and fill missing values with the median.
    3. Drop the `customerID` identifier column.
    4. Engineer two new features:
         - `tenure_group`: bucketed tenure (0-12, 12-24, 24-48, 48-60, 60+ months)
         - `charge_ratio`: MonthlyCharges / TotalCharges
    5. Label-encode every categorical column (including the engineered
       `tenure_group`).
    6. Encode the target column `Churn` as 0/1.

The module can be used either as a library (import `preprocess_data`) or run
directly as a script, in which case it saves the cleaned dataset and the
fitted label encoders to disk for reuse by `model_training.py`.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default source of the dataset (can be a local file path or, as here, a URL).
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_data(filepath_or_url: str = DATA_URL) -> pd.DataFrame:
    """Load the raw Telco churn dataset from a local path or URL.

    Raises:
        RuntimeError: if the file cannot be read for any reason.
    """
    try:
        df = pd.read_csv(filepath_or_url)
        logger.info("Loaded dataset with shape %s from %s", df.shape, filepath_or_url)
        return df
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable error
        raise RuntimeError(f"Failed to load dataset from '{filepath_or_url}': {exc}") from exc


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `TotalCharges` to numeric, filling missing values with the median.

    The raw dataset stores a handful of `TotalCharges` values as blank
    strings (for customers with zero tenure), which pandas would otherwise
    keep as an object/string column.
    """
    df = df.copy()
    if "TotalCharges" not in df.columns:
        raise KeyError("Expected column 'TotalCharges' not found in dataframe.")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    median_value = df["TotalCharges"].median()

    if n_missing:
        logger.info(
            "Filling %d missing TotalCharges value(s) with median %.2f",
            n_missing,
            median_value,
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
    return df


def drop_identifier_columns(df: pd.DataFrame, id_column: str = ID_COLUMN) -> pd.DataFrame:
    """Drop the customer identifier column, which carries no predictive signal."""
    df = df.copy()
    if id_column in df.columns:
        df = df.drop(columns=[id_column])
    return df


def create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket `tenure` (months) into business-friendly groups."""
    df = df.copy()
    if "tenure" not in df.columns:
        raise KeyError("Expected column 'tenure' not found in dataframe.")

    bins = [-np.inf, 12, 24, 48, 60, np.inf]
    labels = ["0-12", "12-24", "24-48", "48-60", "60+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)
    return df


def create_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add `charge_ratio` = MonthlyCharges / TotalCharges.

    Guards against division by zero (customers with zero TotalCharges) by
    setting the ratio to 0 in that edge case rather than propagating NaN/inf.
    """
    df = df.copy()
    safe_denominator = df["TotalCharges"].replace(0, np.nan)
    ratio = df["MonthlyCharges"] / safe_denominator
    df["charge_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def encode_categorical_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Label-encode every categorical (object-dtype) column except the target.

    If `encoders` is provided, reuse those already-fitted encoders instead of
    fitting new ones (used at inference time so new data is mapped onto the
    same encoding learned during training). Unseen categories fall back to
    the encoder's first known class rather than raising an error.

    Returns:
        (encoded_dataframe, encoders_dict)
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    if encoders is None:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            le = encoders.get(col)
            if le is None:
                continue
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])

    return df, encoders


def encode_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Map the target column from Yes/No strings to 1/0 integers."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        df[target_column] = df[target_column].astype(str).map({"Yes": 1, "No": 0})
        if df[target_column].isna().any():
            raise ValueError(
                f"Found values in '{target_column}' outside of the expected {{'Yes', 'No'}}."
            )
    df[target_column] = df[target_column].astype(int)
    return df


def preprocess_data(
    filepath_or_url: str = DATA_URL,
    target_column: str = TARGET_COLUMN,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full preprocessing pipeline end-to-end.

    Args:
        filepath_or_url: source of the raw CSV.
        target_column: name of the churn label column.
        encoders: optional pre-fitted LabelEncoders to reuse (for scoring new
            data with the same encoding used at training time).

    Returns:
        (clean_dataframe, encoders_dict) ready for modelling.
    """
    df = load_data(filepath_or_url)
    df = clean_total_charges(df)
    df = drop_identifier_columns(df)
    df = create_tenure_group(df)
    df = create_charge_ratio(df)
    df, encoders = encode_categorical_columns(df, target_column=target_column, encoders=encoders)
    df = encode_target(df, target_column=target_column)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df, encoders


if __name__ == "__main__":
    clean_df, fitted_encoders = preprocess_data()

    print(clean_df.head())
    print("\nColumn dtypes:")
    print(clean_df.dtypes)

    clean_df.to_csv("processed_churn_data.csv", index=False)
    joblib.dump(fitted_encoders, "label_encoders.pkl")

    print("\nSaved 'processed_churn_data.csv' and 'label_encoders.pkl'.")
