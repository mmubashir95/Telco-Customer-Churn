from __future__ import annotations

import pandas as pd

from src.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

REQUIRED_COLUMNS = [
    "customerID",
    *CATEGORICAL_COLUMNS,
    *NUMERICAL_COLUMNS,
    TARGET_COLUMN,
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
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(f"Target column '{TARGET_COLUMN}' is missing.")

    allowed = {"Yes", "No"}
    actual = set(df[TARGET_COLUMN].dropna().astype(str).str.strip().unique())
    invalid = sorted(actual - allowed)
    if invalid:
        raise DataValidationError(f"Invalid values in '{TARGET_COLUMN}': {invalid}")


def validate_customer_id_unique(df: pd.DataFrame) -> None:
    if "customerID" not in df.columns:
        raise DataValidationError("Identifier column 'customerID' is missing.")

    duplicate_ids = df["customerID"][df["customerID"].duplicated()].unique().tolist()
    if duplicate_ids:
        raise DataValidationError(f"Duplicate customerID values found: {duplicate_ids[:10]}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise DataValidationError(f"Missing values found: {missing.to_dict()}")


def validate_total_charges_numeric(df: pd.DataFrame) -> None:
    if "TotalCharges" not in df.columns:
        raise DataValidationError("Column 'TotalCharges' is missing.")

    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if total_charges.isna().any():
        raise DataValidationError("Column 'TotalCharges' contains non-numeric values.")


def validate_non_negative_numeric_columns(df: pd.DataFrame) -> None:
    numeric_columns = NUMERICAL_COLUMNS
    missing = [column for column in numeric_columns if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing numeric columns for validation: {missing}")

    negatives: dict[str, int] = {}
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        negative_count = int((series < 0).sum())
        if negative_count > 0:
            negatives[column] = negative_count

    if negatives:
        raise DataValidationError(f"Negative values found: {negatives}")


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run all validation checks for raw churn data."""
    validate_non_empty(df)
    validate_required_columns(df)
    validate_churn_values(df)


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """Run validation checks for cleaned churn data."""
    validate_non_empty(df)
    validate_required_columns(df)
    validate_customer_id_unique(df)
    validate_churn_values(df)
    validate_no_missing_values(df)
    validate_total_charges_numeric(df)
    validate_non_negative_numeric_columns(df)
