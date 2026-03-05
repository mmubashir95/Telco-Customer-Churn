# WA_Fn-UseC_-Telco-Customer-Churn

Customer churn analysis project using the IBM Telco dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).

## Problem Statement

Predict whether a telecom customer will churn (`Churn = Yes/No`) using customer demographics, account details, services, and billing behavior.

## Dataset

- File: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Target column: `Churn`
- Columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

## Project Structure

```text
.
├── 01_eda.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── .env
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` is still empty, install at least:

```bash
pip install pandas
```

## Run EDA

```bash
python 01_eda.py
```

Current script behavior:
- Loads dataset with pandas
- Prints dataframe shape

## Next Steps

- Data cleaning (especially `TotalCharges` type/blank handling)
- Exploratory analysis by churn segments
- Feature engineering and encoding
- Baseline and tuned classification models (e.g., Logistic Regression, Random Forest, XGBoost)
- Evaluation with ROC-AUC, F1, precision/recall, confusion matrix
