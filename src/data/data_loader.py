import os
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Telco churn dataset from disk."""
    csv_path = Path(path) if path is not None else DEFAULT_RAW_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    return df
