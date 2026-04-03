from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = "data/raw/telco.csv"
CLEANED_DATA_PATH = "data/interim/telco_churn_cleaned.csv"
ENGINEERED_DATA_PATH = "data/processed/telco_churn_engineered.csv"
SELECTED_DATA_PATH = "data/processed/telco_churn_selected.csv"

TARGET_COLUMN = "Churn"
MODEL_TARGET_COLUMN = "target"

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
ENGINEERED_DATA_FILE = PROJECT_ROOT / ENGINEERED_DATA_PATH
SELECTED_DATA_FILE = PROJECT_ROOT / SELECTED_DATA_PATH

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURE_COLUMNS_FILE = ARTIFACTS_DIR / "feature_columns.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"

