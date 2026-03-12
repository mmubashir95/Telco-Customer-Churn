import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_RELATIVE_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

DEFAULT_RAW_DATA_PATH = Path(
    os.getenv("RAW_DATA_PATH", str(PROJECT_ROOT / DEFAULT_DATA_RELATIVE_PATH))
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
