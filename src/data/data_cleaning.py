from __future__ import annotations

import pandas as pd


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
