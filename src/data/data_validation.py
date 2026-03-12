from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


class DataValidationError(ValueError):
    """Raised when dataset validation fails."""


def validate_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Dataset is empty.")


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_churn_values(df: pd.DataFrame) -> None:
    if "Churn" not in df.columns:
        raise DataValidationError("Target column 'Churn' is missing.")

    allowed = {"Yes", "No"}
    actual = set(df["Churn"].dropna().astype(str).str.strip().unique())
    invalid = sorted(actual - allowed)
    if invalid:
        raise DataValidationError(f"Invalid values in 'Churn': {invalid}")


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run all validation checks for raw churn data."""
    validate_non_empty(df)
    validate_required_columns(df)
    validate_churn_values(df)
