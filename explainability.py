"""
explainability.py
------------------
Generates SHAP-based explanations for the trained churn model.

Steps performed:
    1. Load the trained model from `model.pkl`.
    2. Reconstruct the same held-out test set used during training (same
       preprocessing + same stratified split + random_state) so the
       explanations reflect genuinely unseen data.
    3. Compute SHAP values with `shap.TreeExplainer`.
    4. Save three plots as PNG files:
         - `shap_summary_bar.png`   - top 10 most important features (bar)
         - `shap_beeswarm.png`      - feature impact direction (beeswarm)
         - `shap_waterfall_high_risk.png` - single highest-risk customer
    5. Print the top 5 most important features driving churn.
"""

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_preprocessing import DATA_URL, TARGET_COLUMN, preprocess_data
from model_training import MODEL_PATH, split_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUMMARY_BAR_PATH = "shap_summary_bar.png"
BEESWARM_PATH = "shap_beeswarm.png"
WATERFALL_PATH = "shap_waterfall_high_risk.png"
TOP_N_FEATURES = 5
TOP_N_BAR_PLOT = 10


def load_model(path: str = MODEL_PATH):
    """Load the trained model, with a clear error if training hasn't run yet."""
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find '{path}'. Run `python model_training.py` first "
            "to train and save the model."
        ) from exc


def get_test_set():
    """Rebuild the exact same held-out test split used during training."""
    df, _ = preprocess_data(DATA_URL)
    _, X_test, _, y_test = split_data(df, target_column=TARGET_COLUMN)
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def compute_shap_values(model, X_test: pd.DataFrame):
    """Compute SHAP values for the test set using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def plot_summary_bar(shap_values, X_test: pd.DataFrame, top_n: int = TOP_N_BAR_PLOT, save_path: str = SUMMARY_BAR_PATH) -> None:
    """Save a SHAP bar plot of the top-N most important features."""
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=top_n, show=False)
    plt.title(f"Top {top_n} Features Driving Churn (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary bar plot to '%s'", save_path)


def plot_beeswarm(shap_values, save_path: str = BEESWARM_PATH) -> None:
    """Save a SHAP beeswarm plot showing the direction of each feature's impact."""
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.title("SHAP Feature Impact on Churn Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm plot to '%s'", save_path)


def plot_waterfall_for_high_risk(model, shap_values, X_test: pd.DataFrame, save_path: str = WATERFALL_PATH) -> int:
    """Save a SHAP waterfall plot for the customer with the highest predicted churn probability."""
    probabilities = model.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(probabilities))

    plt.figure()
    shap.plots.waterfall(shap_values[high_risk_idx], show=False)
    plt.title(f"SHAP Waterfall - Customer #{high_risk_idx} (churn probability {probabilities[high_risk_idx]:.1%})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(
        "Saved SHAP waterfall plot for customer index %d (predicted churn probability %.2f%%) to '%s'",
        high_risk_idx,
        probabilities[high_risk_idx] * 100,
        save_path,
    )
    return high_risk_idx


def print_top_features(shap_values, X_test: pd.DataFrame, top_n: int = TOP_N_FEATURES) -> None:
    """Print the top-N features ranked by mean absolute SHAP value."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)

    print("=" * 50)
    print(f"TOP {top_n} FEATURES DRIVING CHURN")
    print("=" * 50)
    for rank, (feature, value) in enumerate(importance.head(top_n).items(), start=1):
        print(f"{rank}. {feature:<20s} mean |SHAP value| = {value:.4f}")
    print("=" * 50)


def main() -> None:
    model = load_model()
    X_test, _ = get_test_set()

    try:
        _, shap_values = compute_shap_values(model, X_test)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to compute SHAP values: %s", exc)
        raise

    plot_summary_bar(shap_values, X_test)
    plot_beeswarm(shap_values)
    plot_waterfall_for_high_risk(model, shap_values, X_test)
    print_top_features(shap_values, X_test)


if __name__ == "__main__":
    main()
