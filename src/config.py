from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = "data/raw/telco.csv"
CLEANED_DATA_PATH = "data/interim/telco_churn_cleaned.csv"
TARGET_COLUMN = "Churn"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_DATA_FILE = PROJECT_ROOT / DATA_PATH
CLEANED_DATA_FILE = PROJECT_ROOT / CLEANED_DATA_PATH
