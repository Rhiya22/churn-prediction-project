# Customer Churn Predictor

A complete, end-to-end machine learning project that predicts customer churn for a telecom company using **XGBoost** for classification and **SHAP** for model explainability, wrapped in an interactive **Streamlit** application.

## Project Overview

Customer churn — when a customer stops doing business with a company — is one of the most expensive problems a subscription-based business faces. This project builds a full pipeline to predict which customers are likely to churn, so retention teams can intervene before it happens.

The pipeline covers the complete lifecycle of a real-world ML project:

- **Data preprocessing** — cleaning the raw dataset and engineering new features.
- **Model training** — an XGBoost classifier tuned for class imbalance, evaluated with standard classification metrics.
- **Explainability** — SHAP values to understand *why* the model makes each prediction, both globally (which features matter most overall) and locally (why one specific customer is flagged as high-risk).
- **Deployment** — an interactive Streamlit app where a user can enter a customer profile and get an instant, explained churn prediction.

## Dataset

This project uses the **IBM Telco Customer Churn** dataset, a widely-used benchmark dataset for churn prediction:

- **Records:** 7,043 customers
- **Features:** 21 columns, including demographic information (gender, senior citizen status, partner/dependents), account information (tenure, contract type, payment method, billing), subscribed services (phone, internet, online security, tech support, streaming), and billing amounts (monthly and total charges)
- **Target:** `Churn` — whether the customer left the company within the last month (`Yes`/`No`)
- **Source:** [IBM Telco Customer Churn on ICP4D](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

## Key Results

The trained XGBoost model achieves the following performance on a held-out 20% test set (stratified split, `random_state=42`):

| Metric    | Score  |
|-----------|--------|
| ROC-AUC   | ~0.84  |
| Precision | ~0.52  |
| Recall    | ~0.77  |
| F1 Score  | ~0.62  |

`scale_pos_weight` was used during training to counteract the class imbalance in the dataset (roughly 27% of customers churn), which biases the model toward higher recall on the churn class — appropriate for a retention use case, where missing an at-risk customer is typically costlier than a false alarm.

SHAP analysis identifies the top drivers of churn as:

1. **Contract type** — month-to-month customers churn far more than those on annual contracts
2. **Charge ratio / Monthly charges** — customers paying more relative to their total spend are more likely to churn
3. **Tenure** — newer customers churn more than long-tenured ones
4. **Tech support** — customers without tech support churn more
5. **Internet service** — fiber optic customers show higher churn than DSL customers

## Tech Stack

- **Python** — core language
- **XGBoost** — gradient-boosted tree classifier
- **SHAP** — model explainability (global and per-prediction)
- **Streamlit** — interactive web application
- **scikit-learn** — train/test splitting and evaluation metrics
- **pandas** / **NumPy** — data manipulation
- **Plotly** — interactive visualizations in the app
- **Matplotlib** / **Seaborn** — SHAP and static plotting

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/customer-churn-predictor.git
cd customer-churn-predictor
```

**2. Install dependencies**

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run preprocessing** (optional standalone step — saves a cleaned CSV and encoders for inspection)

```bash
python data_preprocessing.py
```

**4. Train the model**

```bash
python model_training.py
```

This trains the XGBoost classifier, prints evaluation metrics, and saves `model.pkl`, `label_encoders.pkl`, `feature_columns.pkl`, and `default_values.pkl` to the project root.

**5. (Optional) Generate SHAP explanations**

```bash
python explainability.py
```

This saves three plots (`shap_summary_bar.png`, `shap_beeswarm.png`, `shap_waterfall_high_risk.png`) and prints the top 5 features driving churn.

**6. Launch the Streamlit app**

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`), enter a customer profile in the sidebar, and click **Predict** to see the churn risk and its SHAP-based explanation.

## Project Structure

```
customer-churn-predictor/
├── data_preprocessing.py   # Cleans raw data, engineers features, encodes categoricals
├── model_training.py       # Trains and evaluates the XGBoost model, saves all artifacts
├── explainability.py       # Generates global & local SHAP explanations as PNG plots
├── app.py                  # Streamlit app: sidebar inputs -> prediction -> SHAP explanation
├── requirements.txt        # Pinned Python dependencies
├── .gitignore              # Ignores generated artifacts, data files, caches
└── README.md               # Project documentation (this file)
```

| File | Purpose |
|---|---|
| `data_preprocessing.py` | Loads the raw CSV, fixes `TotalCharges`, drops `customerID`, engineers `tenure_group` and `charge_ratio`, label-encodes categoricals and the target. |
| `model_training.py` | Stratified 80/20 split, trains an imbalance-aware XGBoost classifier, prints ROC-AUC/precision/recall/F1/confusion matrix, saves the model and supporting artifacts with `joblib`. |
| `explainability.py` | Loads the saved model, computes SHAP values on the test set, saves a summary bar plot, a beeswarm plot, and a waterfall plot for the highest-risk customer, and prints the top 5 features. |
| `app.py` | A Streamlit UI for scoring a single customer profile interactively, with a colour-coded risk gauge and a SHAP waterfall explanation for that specific prediction. |

## Screenshots

*(Add screenshots of the running Streamlit app here, e.g. the input sidebar, the risk gauge, and the SHAP waterfall explanation.)*

```
docs/screenshot-input.png
docs/screenshot-prediction.png
docs/screenshot-shap.png
```

## Author

**Rhiya Raman**
