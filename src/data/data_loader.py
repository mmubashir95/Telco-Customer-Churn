import os
from pathlib import Path

import pandas as pd

from src.config import CLEANED_DATA_FILE, PROJECT_ROOT, RAW_DATA_FILE

DEFAULT_RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", str(RAW_DATA_FILE))
)
DEFAULT_CLEANED_DATA_PATH = Path(
    os.getenv("CLEANED_DATA_PATH", str(CLEANED_DATA_FILE))
)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Telco churn dataset from disk."""
    csv_path = Path(path) if path is not None else DEFAULT_RAW_DATA_PATH
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    return df


def load_cleaned_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned Telco churn dataset from disk."""
    csv_path = Path(path) if path is not None else DEFAULT_CLEANED_DATA_PATH
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    return df
