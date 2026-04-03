from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CLEANED_DATA_FILE, PROJECT_ROOT


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize TotalCharges by trimming blanks and coercing to numeric."""
    df_clean = df.copy()
    df_clean["TotalCharges"] = (
        df_clean["TotalCharges"].astype(str).str.strip().replace("", pd.NA)
    )
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean["TotalCharges"], errors="coerce"
    )
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)
    return df_clean


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all reusable cleaning steps for the Telco churn dataset."""
    df_clean = df.copy()
    df_clean = clean_total_charges(df_clean)
    return df_clean


def save_cleaned_data(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """Persist the cleaned dataset to the interim data folder."""
    output_path = Path(path) if path is not None else CLEANED_DATA_FILE
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
