"""
app.py
------
Streamlit web application for the Customer Churn Predictor — reimagined as a
retention team's working tool rather than a single-customer data-entry form.

Interaction model:
    1. Header shows the model's overall performance (ROC-AUC, recall) on the
       held-out test set, so the tool establishes credibility up front.
    2. Main view is a churn-risk table: real (held-out, unseen) customers
       ranked by predicted churn probability, highest risk first — the kind
       of worklist a retention team would actually triage from.
    3. Selecting a customer opens a detail view: their key attributes, a
       SHAP waterfall explaining *their* specific risk, and a short
       natural-language summary of the top drivers.
    4. A "What if?" panel lets the user edit a couple of that customer's
       attributes (contract, tech support, monthly charges) and see the
       churn probability recompute live — the retention-relevant question of
       "if we intervened on X, does the risk drop?"

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    DATA_URL,
    TARGET_COLUMN,
    clean_total_charges,
    create_charge_ratio,
    create_tenure_group,
    encode_categorical_columns,
    encode_target,
    load_data,
)
from model_training import RANDOM_STATE, TEST_SIZE

MODEL_PATH = "model.pkl"
ENCODERS_PATH = "label_encoders.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
DEFAULT_VALUES_PATH = "default_values.pkl"

GITHUB_URL = "https://github.com/Rhiya22"

TOP_N_CUSTOMERS = 50  # size of the retention team's risk worklist

# Friendly labels for feature names, used in the driver summary and detail view.
FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "tenure_group": "Tenure group",
    "charge_ratio": "Charge ratio (monthly / total)",
}

st.set_page_config(
    page_title="Churn Risk Table",
    page_icon="\U0001F4C9",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# Artifact loading (self-training fallback: trains on first run if missing)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Load the trained model and supporting artifacts, cached across reruns.

    If the model hasn't been trained yet in this environment (e.g. right
    after a fresh deployment, where the .pkl files aren't in the repo on
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
        st.error(f"Model artifacts still weren't found after training. Details: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading model artifacts: {exc}")
        st.stop()


@st.cache_data(show_spinner="Loading customer risk table...")
def load_test_set(_encoders: dict, feature_columns: list):
    """Rebuild the exact held-out test split used during training/evaluation.

    Returns a tuple of:
        display_df  - the test customers with their original, human-readable
                       values (customerID, "Month-to-month", etc.)
        model_df    - the same rows, fully encoded in the schema the model
                       expects (same column order as `feature_columns`)
        y_test      - true churn labels (0/1) for the same rows

    The leading underscore on `_encoders` tells Streamlit not to try to hash
    the (unhashable) LabelEncoder objects when deciding whether to reuse the
    cache.
    """
    raw = load_data(DATA_URL)
    raw = clean_total_charges(raw)
    raw = create_tenure_group(raw)
    raw = create_charge_ratio(raw)

    model_df = raw.drop(columns=["customerID"]).copy()
    model_df, _ = encode_categorical_columns(model_df, target_column=TARGET_COLUMN, encoders=_encoders)
    model_df = encode_target(model_df, target_column=TARGET_COLUMN)

    y_full = model_df[TARGET_COLUMN]
    idx_train, idx_test = train_test_split(
        raw.index, test_size=TEST_SIZE, stratify=y_full, random_state=RANDOM_STATE
    )

    display_df = raw.loc[idx_test].reset_index(drop=True)
    model_df = model_df.loc[idx_test, feature_columns].reset_index(drop=True)
    y_test = y_full.loc[idx_test].reset_index(drop=True)

    return display_df, model_df, y_test


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    """Cache the SHAP TreeExplainer so it's built once per model, not per rerun."""
    return shap.TreeExplainer(_model)


# --------------------------------------------------------------------------- #
# Encoding helpers (mirrors data_preprocessing.py's encoding at inference time)
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


def build_model_row(raw_row: dict, encoders: dict, feature_columns: list) -> pd.DataFrame:
    """Encode a raw (human-readable) customer row into the model's schema."""
    row = dict(raw_row)

    total_charges = row.get("TotalCharges")
    if total_charges in (0, None) or (isinstance(total_charges, float) and np.isnan(total_charges)):
        charge_ratio = 0.0
    else:
        charge_ratio = row["MonthlyCharges"] / total_charges
    row["charge_ratio"] = 0.0 if (charge_ratio is None or np.isnan(charge_ratio)) else charge_ratio

    encoded = {}
    for col in feature_columns:
        value = row.get(col)
        encoded[col] = encode_value(encoders, col, value) if col in encoders else value
    return pd.DataFrame([encoded])[feature_columns]


def risk_tier(probability: float) -> tuple[str, str]:
    """Map a churn probability to a (label, colour) risk tier."""
    if probability < 0.30:
        return "Low", "#2ecc71"
    if probability < 0.70:
        return "Medium", "#f39c12"
    return "High", "#e74c3c"


def describe_top_drivers(shap_row: pd.Series, raw_row: pd.Series, top_n: int = 3) -> list[str]:
    """Turn a customer's SHAP values into short natural-language sentences."""
    ordered_features = shap_row.abs().sort_values(ascending=False).index[:top_n]
    sentences = []
    for feat in ordered_features:
        impact = shap_row[feat]
        direction = "raises" if impact > 0 else "lowers"
        friendly = FEATURE_LABELS.get(feat, feat)
        raw_value = raw_row.get(feat, None)
        if isinstance(raw_value, float) and raw_value == int(raw_value):
            raw_value = int(raw_value)
        sentences.append(
            f"**{friendly}** (currently *{raw_value}*) **{direction}** this customer's churn risk "
            f"(SHAP impact {impact:+.3f})."
        )
    return sentences


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_header(y_test: pd.Series, probabilities: np.ndarray) -> None:
    st.title("Churn Risk Table")
    st.caption("A retention-team worklist: which customers are most likely to churn, why, and what might change it.")

    auc = roc_auc_score(y_test, probabilities)
    recall = recall_score(y_test, (probabilities >= 0.5).astype(int), zero_division=0)
    churn_rate = float(y_test.mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model ROC-AUC", f"{auc:.2f}", help="Measured on the held-out test set (unseen during training).")
    c2.metric("Recall (catches actual churners)", f"{recall:.0%}")
    c3.metric("Test-set customers", f"{len(y_test):,}")
    c4.metric("Historical churn rate", f"{churn_rate:.0%}")
    st.divider()


def render_leaderboard(display_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Render the sortable, ranked risk table and return it (with risk column)."""
    table = display_df.copy()
    table["Churn Risk"] = probabilities
    table = table.sort_values("Churn Risk", ascending=False).head(TOP_N_CUSTOMERS).reset_index(drop=True)
    table.insert(0, "Rank", np.arange(1, len(table) + 1))

    st.subheader(f"Top {len(table)} at-risk customers")
    st.caption("Ranked highest risk first. Click a column header to re-sort; pick a customer below to inspect them.")

    display_cols = [
        "Rank",
        "customerID",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "InternetService",
        "TechSupport",
        "Churn Risk",
    ]
    st.dataframe(
        table[display_cols],
        width="stretch",
        hide_index=True,
        height=430,
        column_config={
            "customerID": st.column_config.TextColumn("Customer ID"),
            "Contract": st.column_config.TextColumn("Contract"),
            "tenure": st.column_config.NumberColumn("Tenure (mo)"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
            "InternetService": st.column_config.TextColumn("Internet"),
            "TechSupport": st.column_config.TextColumn("Tech Support"),
            "Churn Risk": st.column_config.ProgressColumn(
                "Churn Risk", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
        },
    )
    return table


def render_detail_and_whatif(
    table: pd.DataFrame,
    model_features_test: pd.DataFrame,
    display_df: pd.DataFrame,
    model,
    encoders: dict,
    feature_columns: list,
) -> None:
    st.divider()
    st.subheader("Customer detail & explanation")

    customer_ids = table["customerID"].tolist()
    selected_id = st.selectbox(
        "Select a customer from the table above to inspect",
        options=customer_ids,
        index=0,
    )

    selected_rank_row = table.loc[table["customerID"] == selected_id].iloc[0]
    original_index = display_df.index[display_df["customerID"] == selected_id][0]
    raw_row = display_df.loc[original_index]
    model_row = model_features_test.loc[[original_index]][feature_columns]
    original_probability = float(model.predict_proba(model_row)[:, 1][0])
    label, color = risk_tier(original_probability)

    col_info, col_shap = st.columns([1, 1.3])

    with col_info:
        st.markdown(f"#### Customer `{selected_id}`")
        st.markdown(f"<h3 style='color:{color};'>{label} risk - {original_probability:.1%}</h3>", unsafe_allow_html=True)
        attrs = {
            "Contract": raw_row["Contract"],
            "Tenure": f"{int(raw_row['tenure'])} months",
            "Monthly charges": f"${raw_row['MonthlyCharges']:.2f}",
            "Total charges": f"${raw_row['TotalCharges']:.2f}",
            "Internet service": raw_row["InternetService"],
            "Tech support": raw_row["TechSupport"],
            "Online security": raw_row["OnlineSecurity"],
            "Payment method": raw_row["PaymentMethod"],
        }
        st.table(pd.DataFrame(attrs.items(), columns=["Attribute", "Value"]).set_index("Attribute"))

    explainer = get_explainer(model)
    shap_values = explainer(model_row)

    with col_shap:
        st.caption("Why this customer is flagged: SHAP waterfall for their specific prediction.")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    shap_row = pd.Series(shap_values.values[0], index=feature_columns)
    st.markdown("**Top drivers of this customer's risk:**")
    for sentence in describe_top_drivers(shap_row, raw_row):
        st.markdown(f"- {sentence}")

    render_whatif_panel(selected_id, raw_row, original_probability, model, encoders, feature_columns)


def render_whatif_panel(
    selected_id: str,
    raw_row: pd.Series,
    original_probability: float,
    model,
    encoders: dict,
    feature_columns: list,
) -> None:
    st.divider()
    st.subheader("What if we intervened?")
    st.caption(
        "Try changing this customer's contract, tech support, or monthly charge to see whether a plausible "
        "retention offer would actually move their predicted risk."
    )

    contract_options = list(encoders["Contract"].classes_)
    techsupport_options = list(encoders["TechSupport"].classes_)
    has_internet = raw_row["InternetService"] != "No"

    c1, c2, c3 = st.columns(3)
    with c1:
        whatif_contract = st.selectbox(
            "Contract type",
            options=contract_options,
            index=contract_options.index(raw_row["Contract"]),
            key=f"whatif_contract_{selected_id}",
        )
    with c2:
        if has_internet:
            techsupport_choices = [c for c in techsupport_options if c != "No internet service"]
            whatif_techsupport = st.selectbox(
                "Tech support",
                options=techsupport_choices,
                index=techsupport_choices.index(raw_row["TechSupport"])
                if raw_row["TechSupport"] in techsupport_choices
                else 0,
                key=f"whatif_techsupport_{selected_id}",
            )
        else:
            whatif_techsupport = raw_row["TechSupport"]
            st.selectbox(
                "Tech support",
                options=["No internet service"],
                index=0,
                disabled=True,
                key=f"whatif_techsupport_disabled_{selected_id}",
                help="This customer has no internet service, so tech support doesn't apply.",
            )
    with c3:
        whatif_monthly = st.slider(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=float(raw_row["MonthlyCharges"]),
            step=1.0,
            key=f"whatif_monthly_{selected_id}",
        )

    whatif_row = dict(raw_row)
    whatif_row["Contract"] = whatif_contract
    whatif_row["TechSupport"] = whatif_techsupport
    whatif_row["MonthlyCharges"] = whatif_monthly

    whatif_model_row = build_model_row(whatif_row, encoders, feature_columns)
    whatif_probability = float(model.predict_proba(whatif_model_row)[:, 1][0])
    delta = whatif_probability - original_probability

    r1, r2, r3 = st.columns(3)
    r1.metric("Current predicted risk", f"{original_probability:.1%}")
    r2.metric(
        "Risk with this scenario",
        f"{whatif_probability:.1%}",
        delta=f"{delta:+.1%}",
        delta_color="inverse",
    )
    if delta < -0.01:
        r3.success(f"This scenario lowers churn risk by {abs(delta):.1%}.")
    elif delta > 0.01:
        r3.warning(f"This scenario raises churn risk by {delta:.1%}.")
    else:
        r3.info("This scenario barely moves the predicted risk.")


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
    model, encoders, feature_columns, _defaults = load_artifacts()
    display_df, model_df, y_test = load_test_set(encoders, feature_columns)

    probabilities = model.predict_proba(model_df[feature_columns])[:, 1]

    render_header(y_test, probabilities)
    table = render_leaderboard(display_df, probabilities)
    render_detail_and_whatif(table, model_df, display_df, model, encoders, feature_columns)
    render_footer()


if __name__ == "__main__":
    main()
