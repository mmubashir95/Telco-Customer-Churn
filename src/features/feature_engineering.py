from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif

from src.config import (
    CATEGORICAL_COLUMNS,
    MODEL_TARGET_COLUMN,
    NUMERICAL_COLUMNS,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_TABLES_DIR,
    TARGET_COLUMN,
)


def engineer_telco_features(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    output_target_column: str = MODEL_TARGET_COLUMN,
) -> pd.DataFrame:
    """Create the engineered churn feature set used in notebook 09."""
    feature_df = df.copy()

    if "TotalCharges" in feature_df.columns:
        feature_df["log_total_charges"] = np.log1p(feature_df["TotalCharges"].fillna(0))
        feature_df = feature_df.drop(columns=["TotalCharges"])

    if "tenure" in feature_df.columns:
        feature_df["tenure_band"] = pd.cut(
            feature_df["tenure"],
            bins=[-0.1, 12, 24, 48, 72],
            labels=["new", "early", "mid", "long_term"],
        )

    binary_feature_mappings = {
        "gender": {"Female": 0, "Male": 1},
        "Partner": {"No": 0, "Yes": 1},
        "Dependents": {"No": 0, "Yes": 1},
        "PhoneService": {"No": 0, "Yes": 1},
        "PaperlessBilling": {"No": 0, "Yes": 1},
    }
    for column, mapping in binary_feature_mappings.items():
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].map(mapping).astype("Int64")

    if "SeniorCitizen" in feature_df.columns:
        feature_df["SeniorCitizen"] = feature_df["SeniorCitizen"].astype("Int64")

    if target_column in feature_df.columns:
        feature_df[target_column] = (
            feature_df[target_column].map({"No": 0, "Yes": 1}).astype("Int64")
        )
        feature_df = feature_df.rename(columns={target_column: output_target_column})

    if "Contract" in feature_df.columns:
        feature_df["Contract_ordinal"] = feature_df["Contract"].map(
            {"Month-to-month": 0, "One year": 1, "Two year": 2}
        ).astype("Int64")

    if "PaymentMethod" in feature_df.columns:
        feature_df["payment_method_group"] = feature_df["PaymentMethod"].replace(
            {
                "Bank transfer (automatic)": "auto_payment",
                "Credit card (automatic)": "auto_payment",
                "Electronic check": "electronic_check",
                "Mailed check": "manual_check",
            }
        )

    if "InternetService" in feature_df.columns:
        feature_df["internet_service_group"] = feature_df["InternetService"].replace(
            {
                "Fiber optic": "fiber",
                "DSL": "dsl",
                "No": "no_internet",
            }
        )

    if {"tenure", "MonthlyCharges"}.issubset(feature_df.columns):
        feature_df["tenure_x_MonthlyCharges"] = (
            feature_df["tenure"] * feature_df["MonthlyCharges"]
        )

    if {"Contract", "PaymentMethod"}.issubset(feature_df.columns):
        feature_df["contract_payment_profile"] = (
            feature_df["Contract"].astype(str)
            + "__"
            + feature_df["PaymentMethod"].astype(str)
        )

    if {"InternetService", "TechSupport"}.issubset(feature_df.columns):
        feature_df["internet_techsupport_profile"] = (
            feature_df["InternetService"].astype(str)
            + "__"
            + feature_df["TechSupport"].astype(str)
        )

    if {"InternetService", "OnlineSecurity"}.issubset(feature_df.columns):
        feature_df["internet_onlinesecurity_profile"] = (
            feature_df["InternetService"].astype(str)
            + "__"
            + feature_df["OnlineSecurity"].astype(str)
        )

    if {"log_total_charges", "tenure"}.issubset(feature_df.columns):
        total_charges_from_log = np.expm1(feature_df["log_total_charges"])
        tenure_nonzero = feature_df["tenure"].replace(0, np.nan)
        feature_df["avg_monthly_spend"] = (
            total_charges_from_log / tenure_nonzero
        ).fillna(feature_df["MonthlyCharges"])

    if "tenure" in feature_df.columns:
        feature_df["is_new_customer"] = (feature_df["tenure"] < 12).astype(int)

    support_source_columns = [
        column
        for column in ["TechSupport", "OnlineSecurity", "OnlineBackup"]
        if column in feature_df.columns
    ]
    if support_source_columns:
        feature_df["has_support_services"] = (
            feature_df[support_source_columns].eq("Yes").any(axis=1)
        ).astype(int)

    if "Contract" in feature_df.columns:
        feature_df["is_month_to_month"] = (
            feature_df["Contract"] == "Month-to-month"
        ).astype(int)

    if "PaymentMethod" in feature_df.columns:
        feature_df["is_auto_payment"] = feature_df["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        ).astype(int)

    service_count_columns = [
        column
        for column in [
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        if column in feature_df.columns
    ]
    if service_count_columns:
        feature_df["service_count"] = feature_df[service_count_columns].eq("Yes").sum(
            axis=1
        )

    return feature_df


def build_feature_selection_support(
    feature_df: pd.DataFrame,
    *,
    target_column: str = MODEL_TARGET_COLUMN,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build notebook-style correlation, chi-square, and mutual-information support."""
    selection_df = feature_df.copy()
    for column in selection_df.select_dtypes(include=["object", "category"]).columns:
        if column != target_column:
            selection_df[column] = (
                selection_df[column].astype("category").cat.codes
            )

    y = selection_df[target_column].astype(int)
    drop_columns = [
        column
        for column in [target_column, "customerID"]
        if column in selection_df.columns
    ]
    X = selection_df.drop(columns=drop_columns)

    correlation_rows = []
    for column in X.columns:
        correlation_value = pd.Series(X[column]).corr(y)
        correlation_rows.append(
            {
                "feature": column,
                "target_correlation": round(float(correlation_value), 4),
                "abs_target_correlation": round(abs(float(correlation_value)), 4),
            }
        )
    correlation_rank_df = pd.DataFrame(correlation_rows).sort_values(
        "abs_target_correlation",
        ascending=False,
    )

    chi2_candidate_columns = [
        column
        for column in [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "SeniorCitizen",
            "Contract_ordinal",
            "payment_method_group",
            "internet_service_group",
            "contract_payment_profile",
            "internet_techsupport_profile",
            "internet_onlinesecurity_profile",
            "is_new_customer",
            "has_support_services",
            "is_month_to_month",
            "is_auto_payment",
            "service_count",
        ]
        if column in X.columns
    ]

    chi2_rank_df = pd.DataFrame(columns=["feature", "chi2_score", "chi2_p_value"])
    if chi2_candidate_columns:
        chi2_input = X[chi2_candidate_columns].copy()
        chi2_scores, chi2_p_values = chi2(chi2_input.clip(lower=0), y)
        chi2_rank_df = pd.DataFrame(
            {
                "feature": chi2_input.columns,
                "chi2_score": np.round(chi2_scores, 4),
                "chi2_p_value": chi2_p_values,
            }
        ).sort_values("chi2_score", ascending=False)

    mutual_info_scores = mutual_info_classif(X, y, random_state=random_state)
    mutual_info_rank_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mutual_information": np.round(mutual_info_scores, 4),
        }
    ).sort_values("mutual_information", ascending=False)

    return (
        correlation_rank_df.merge(chi2_rank_df, on="feature", how="outer")
        .merge(mutual_info_rank_df, on="feature", how="outer")
        .sort_values(
            ["mutual_information", "abs_target_correlation"],
            ascending=[False, False],
        )
    )


def export_engineered_dataset(
    feature_df: pd.DataFrame,
    *,
    target_column: str = MODEL_TARGET_COLUMN,
    report_dir: Path | None = None,
    processed_path: Path | None = None,
) -> dict[str, object]:
    """Export the engineered dataset and core notebook 09 report tables."""
    feature_groups = {
        "target_column": target_column,
        "numerical_columns": [
            column for column in NUMERICAL_COLUMNS if column in feature_df.columns
        ],
        "categorical_columns": [
            column for column in CATEGORICAL_COLUMNS if column in feature_df.columns
        ],
        "shape": feature_df.shape,
    }
    final_engineered_df = feature_df.drop(
        columns=[column for column in ["customerID"] if column in feature_df.columns]
    ).copy()
    feature_columns = [
        column for column in final_engineered_df.columns if column != target_column
    ]
    feature_selection_support_df = build_feature_selection_support(
        final_engineered_df,
        target_column=target_column,
    )

    report_dir = (
        REPORTS_TABLES_DIR / "feature_engineering" if report_dir is None else report_dir
    )
    processed_path = (
        PROJECT_ROOT / "data" / "processed" / "telco_churn_engineered.csv"
        if processed_path is None
        else processed_path
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    report_dataset_path = report_dir / "feature_engineered_dataset.csv"
    feature_list_path = report_dir / "feature_list.csv"
    feature_groups_path = report_dir / "feature_groups.csv"
    feature_selection_support_path = report_dir / "feature_selection_support.csv"

    final_engineered_df.to_csv(report_dataset_path, index=False)
    final_engineered_df.to_csv(processed_path, index=False)
    pd.Series(feature_columns, name="feature").to_csv(feature_list_path, index=False)
    pd.DataFrame([feature_groups]).to_csv(feature_groups_path, index=False)
    feature_selection_support_df.to_csv(feature_selection_support_path, index=False)

    return {
        "dataset_shape": final_engineered_df.shape,
        "feature_count": len(feature_columns),
        "target_column": target_column,
        "report_dataset_path": str(report_dataset_path.relative_to(PROJECT_ROOT)),
        "processed_dataset_path": str(processed_path.relative_to(PROJECT_ROOT)),
    }

