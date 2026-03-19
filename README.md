# WA_Fn-UseC_-Telco-Customer-Churn

Customer churn analysis project using the IBM Telco dataset.

## Problem Statement

Predict whether a telecom customer will churn (`Churn = Yes/No`) using customer demographics, account details, services, and billing behavior.

## Dataset

- File: `data/raw/telco.csv`
- Target column: `Churn`
- Columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

## Project Structure

```text
.
├── 01_eda.py
├── src/
│   ├── config.py
│   └── data/
│       ├── data_cleaning.py
│       ├── data_loader.py
│       └── data_validation.py
├── data/
│   ├── raw/
│   │   ├── telco.csv
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── interim/
│   │   └── telco_churn_cleaned.csv
│   ├── processed/
│   └── external/
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

## Data Path Configuration

Core project settings live in `src/config.py`.

Default paths are:

```python
DATA_PATH = "data/raw/telco.csv"
CLEANED_DATA_PATH = "data/interim/telco_churn_cleaned.csv"
TARGET_COLUMN = "Churn"
```

Raw and cleaned dataset paths can still be overridden with environment variables:

```env
RAW_DATA_PATH=data/raw/telco.csv
CLEANED_DATA_PATH=data/interim/telco_churn_cleaned.csv
```

## Load Data in Code

```python
from src.data.data_loader import load_raw_data

df = load_raw_data()
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
