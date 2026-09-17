"""
app.py
------
Streamlit web application for the Customer Churn Predictor.

Loads the trained XGBoost model plus its supporting artifacts (label
encoders, feature column order, default feature values) and lets a user
enter a handful of key customer attributes in the sidebar. On clicking
"Predict", it builds a full feature row (filling anything not collected in
the sidebar with dataset-derived defaults), scores it with the model, and
shows:
    - the churn probability as a colour-coded risk gauge (Plotly)
    - a SHAP waterfall chart explaining the individual prediction

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

MODEL_PATH = "model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
DEFAULT_VALUES_PATH = "default_values.pkl"

GITHUB_URL = "https://github.com/Rhiya22"

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="\U0001F4C9",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Load the trained model and supporting artifacts, cached across reruns.

    If the model hasn't been trained yet in this environment (e.g. right
    after a fresh deployment, where `model.pkl` isn't in the repo on
    purpose - see .gitignore), this trains it on the fly so the app is
    fully self-contained and never requires a manual
    `python model_training.py` step before it can be used.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First-time setup: training the model (only happens once per deployment)..."):
            try:
                from model_training import main as train_and_save_artifacts

                train_and_save_artifacts()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Automatic model training failed: {exc}")
                st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        defaults = joblib.load(DEFAULT_VALUES_PATH)
        return model, encoders, feature_columns, defaults
    except FileNotFoundError as exc:
        st.error(
            "Model artifacts still weren't found after training. "
            f"Details: {exc}"
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading model artifacts: {exc}")
        st.stop()


# --------------------------------------------------------------------------- #
# Feature engineering / encoding (mirrors data_preprocessing.py)
# --------------------------------------------------------------------------- #
def encode_value(encoders: dict, column: str, value):
    """Encode a single categorical value with a fitted LabelEncoder.

    Falls back to the encoder's first known class if the value was never
    seen during training, so the app never crashes on an unexpected input.
    """
    le = encoders.get(column)
    if le is None:
        return value
    if value not in le.classes_:
        value = le.classes_[0]
    return int(le.transform([value])[0])


def compute_tenure_group(tenure: float) -> str:
    """Bucket tenure into the same groups used at training time."""
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "12-24"
    if tenure <= 48:
        return "24-48"
    if tenure <= 60:
        return "48-60"
    return "60+"


def build_feature_row(
    user_inputs: dict,
    encoders: dict,
    feature_columns: list,
    defaults: dict,
) -> pd.DataFrame:
    """Assemble a single-row, fully-encoded dataframe matching the training schema.

    Any feature not collected via the sidebar is filled with its dataset
    median/mode (computed in `model_training.py`), so the model always
    receives a complete, correctly-ordered feature vector.
    """
    row = dict(defaults)
    row.update(user_inputs)

    row["tenure_group"] = compute_tenure_group(row["tenure"])

    total_charges = row["TotalCharges"] if row["TotalCharges"] not in (0, None) else np.nan
    charge_ratio = row["MonthlyCharges"] / total_charges if total_charges and not np.isnan(total_charges) else 0.0
    row["charge_ratio"] = 0.0 if np.isnan(charge_ratio) else charge_ratio

    encoded_row = {}
    for col in feature_columns:
        value = row.get(col, defaults.get(col, 0))
        encoded_row[col] = encode_value(encoders, col, value) if col in encoders else value

    return pd.DataFrame([encoded_row])[feature_columns]


def risk_level(probability: float) -> tuple[str, str]:
    """Map a churn probability to a (label, colour) risk tier."""
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk", "#2ecc71"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium Risk", "#f39c12"
    return "High Risk", "#e74c3c"


# --------------------------------------------------------------------------- #
# UI components
# --------------------------------------------------------------------------- #
def render_probability_gauge(probability: float, color: str) -> go.Figure:
    """Build an interactive Plotly gauge for the predicted churn probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Predicted Churn Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#eafaf1"},
                    {"range": [30, 70], "color": "#fef5e7"},
                    {"range": [70, 100], "color": "#fdedec"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.8,
                    "value": round(probability * 100, 1),
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def render_shap_waterfall(model, feature_row: pd.DataFrame) -> plt.Figure:
    """Compute and render a SHAP waterfall plot for a single prediction."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(feature_row)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig


def sidebar_inputs() -> dict:
    """Render the sidebar controls and return the collected customer inputs."""
    st.sidebar.header("Customer Details")

    tenure = st.sidebar.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly_charges = st.sidebar.slider(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
    )
    suggested_total = round(tenure * monthly_charges, 2)
    total_charges = st.sidebar.slider(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=float(min(suggested_total, 10000.0)),
        step=10.0,
        help="Defaults to tenure x monthly charges; adjust if needed.",
    )
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])

    return {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
    }


def render_header() -> None:
    st.title("Customer Churn Predictor")
    st.markdown("##### Powered by XGBoost and SHAP Explainability")
    st.divider()


def render_footer() -> None:
    st.divider()
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; padding-top: 8px; font-size: 0.85rem;'>
            Built by <strong>Rhiya Raman</strong> &nbsp;|&nbsp;
            <a href="{GITHUB_URL}" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
def main() -> None:
    render_header()
    model, encoders, feature_columns, defaults = load_artifacts()

    user_inputs = sidebar_inputs()
    predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

    if not predict_clicked:
        st.info("Adjust the customer details in the sidebar and click **Predict** to see the churn risk.")
        render_footer()
        return

    try:
        feature_row = build_feature_row(user_inputs, encoders, feature_columns, defaults)
        probability = float(model.predict_proba(feature_row)[:, 1][0])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        render_footer()
        return

    label, color = risk_level(probability)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(render_probability_gauge(probability, color), use_container_width=True)
    with col2:
        st.markdown("### Result")
        st.markdown(f"<h2 style='color:{color};'>{label} — {probability:.1%}</h2>", unsafe_allow_html=True)
        st.write("Estimated churn risk based on the customer profile provided in the sidebar.")
        with st.expander("View input profile"):
            profile_df = pd.DataFrame([user_inputs]).T.rename(columns={0: "Value"}).astype(str)
            st.dataframe(profile_df, use_container_width=True)

    st.divider()
    st.markdown("### Why this prediction?")
    st.caption("SHAP waterfall plot showing which features pushed this prediction toward or away from churn.")
    try:
        fig = render_shap_waterfall(model, feature_row)
        st.pyplot(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not generate SHAP explanation: {exc}")

    render_footer()


if __name__ == "__main__":
    main()
