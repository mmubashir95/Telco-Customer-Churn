from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS_FILE,
    MODEL_TARGET_COLUMN,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_TABLES_DIR,
    SELECTED_DATA_FILE,
    TEST_SIZE,
)

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


def _compute_vif_fallback(vif_input: pd.DataFrame) -> pd.DataFrame:
    vif_rows = []
    for feature in vif_input.columns:
        predictors = vif_input.drop(columns=[feature])
        if predictors.shape[1] == 0:
            vif_value = 1.0
        else:
            target = vif_input[feature]
            reg = LinearRegression()
            reg.fit(predictors, target)
            r_squared = reg.score(predictors, target)
            vif_value = np.inf if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)
        vif_rows.append({"feature": feature, "vif": round(float(vif_value), 4)})
    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def _prepare_encoded_feature_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, object]:
    raw_numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    binary_features = [
        column
        for column in raw_numeric_features
        if set(pd.Series(X_train[column]).dropna().astype(float).unique()).issubset({0.0, 1.0})
    ]
    numerical_features = [column for column in raw_numeric_features if column not in binary_features]
    all_object_features = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    redundant_originals = [
        column for column in all_object_features if f"{column}_ordinal" in X_train.columns
    ]
    encoded_categorical_features = [
        column for column in all_object_features if column not in redundant_originals
    ]

    X_train_clean = X_train.drop(columns=redundant_originals, errors="ignore")
    X_test_clean = X_test.drop(columns=redundant_originals, errors="ignore")

    X_train_encoded = pd.get_dummies(
        X_train_clean,
        columns=encoded_categorical_features,
        drop_first=False,
        dtype=int,
    )
    X_test_encoded = pd.get_dummies(
        X_test_clean,
        columns=encoded_categorical_features,
        drop_first=False,
        dtype=int,
    )
    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded,
        join="left",
        axis=1,
        fill_value=0,
    )

    feature_type_map: dict[str, str] = {}
    for feature in numerical_features:
        feature_type_map[feature] = "numerical"
    for feature in binary_features:
        feature_type_map[feature] = "binary"
    for feature in encoded_categorical_features:
        feature_type_map[feature] = "encoded_categorical"
    for feature in X_train_encoded.columns:
        if feature not in feature_type_map:
            parent_feature = next(
                (
                    column
                    for column in encoded_categorical_features
                    if feature.startswith(f"{column}_")
                ),
                None,
            )
            feature_type_map[feature] = (
                "encoded_categorical_dummy" if parent_feature else "derived_numeric"
            )

    feature_variance = X_train_encoded.var(numeric_only=True)
    constant_features = feature_variance[feature_variance == 0].index.tolist()
    near_constant_threshold = 0.01
    near_constant_features = feature_variance[
        (feature_variance > 0) & (feature_variance <= near_constant_threshold)
    ].sort_values().index.tolist()

    X_train_selection = X_train_encoded.drop(columns=constant_features, errors="ignore").copy()
    X_test_selection = X_test_encoded.drop(columns=constant_features, errors="ignore").copy()

    return {
        "numerical_features": numerical_features,
        "binary_features": binary_features,
        "encoded_categorical_features": encoded_categorical_features,
        "redundant_originals": redundant_originals,
        "feature_type_map": feature_type_map,
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "near_constant_threshold": near_constant_threshold,
        "X_train_selection": X_train_selection,
        "X_test_selection": X_test_selection,
    }


def run_feature_selection(
    df: pd.DataFrame,
    *,
    target_column: str = MODEL_TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Run the reusable notebook 10 feature-selection workflow."""
    if target_column not in df.columns:
        raise KeyError(f"{target_column!r} was not found in the dataset.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    encoded = _prepare_encoded_feature_matrices(X_train, X_test)
    numerical_features = encoded["numerical_features"]
    binary_features = encoded["binary_features"]
    feature_type_map = encoded["feature_type_map"]
    X_train_selection = encoded["X_train_selection"]
    X_test_selection = encoded["X_test_selection"]

    target_relationship_rows = []
    for feature in numerical_features + binary_features:
        if feature not in X_train.columns:
            continue
        series = X_train[feature].astype(float)
        pearson_r = series.corr(y_train)
        point_r, point_p = pointbiserialr(y_train, series)
        target_relationship_rows.append(
            {
                "feature": feature,
                "pearson_target_corr": round(float(pearson_r), 4),
                "abs_pearson_target_corr": round(abs(float(pearson_r)), 4),
                "pointbiserial_r": round(float(point_r), 4),
                "pointbiserial_p_value": float(point_p),
            }
        )
    target_relationship_df = pd.DataFrame(target_relationship_rows).sort_values(
        "abs_pearson_target_corr",
        ascending=False,
    ).reset_index(drop=True)

    numeric_corr_matrix = X_train_selection.corr(numeric_only=True)
    high_corr_pairs = []
    for left_index, left_feature in enumerate(numeric_corr_matrix.columns):
        for right_feature in numeric_corr_matrix.columns[left_index + 1 :]:
            corr_value = numeric_corr_matrix.loc[left_feature, right_feature]
            if abs(corr_value) >= 0.7:
                high_corr_pairs.append(
                    {
                        "feature_1": left_feature,
                        "feature_2": right_feature,
                        "correlation": round(float(corr_value), 4),
                        "abs_correlation": round(abs(float(corr_value)), 4),
                    }
                )
    high_corr_pairs_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_pairs_df.empty:
        high_corr_pairs_df = high_corr_pairs_df.sort_values(
            "abs_correlation",
            ascending=False,
        ).reset_index(drop=True)
    else:
        high_corr_pairs_df = pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
        )

    discrete_mask = np.array(
        [
            feature_type_map.get(feature) in {"binary", "encoded_categorical_dummy"}
            for feature in X_train_selection.columns
        ]
    )
    mi_scores = mutual_info_classif(
        X_train_selection,
        y_train,
        discrete_features=discrete_mask,
        random_state=random_state,
    )
    mi_df = pd.DataFrame(
        {
            "feature": X_train_selection.columns,
            "mutual_information": np.round(mi_scores, 4),
        }
    ).sort_values("mutual_information", ascending=False).reset_index(drop=True)

    rf_selector = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_selector.fit(X_train_selection, y_train)
    rf_importance_df = pd.DataFrame(
        {
            "feature": X_train_selection.columns,
            "tree_importance": np.round(rf_selector.feature_importances_, 6),
        }
    ).sort_values("tree_importance", ascending=False).reset_index(drop=True)

    vif_features = [feature for feature in numerical_features if feature in X_train.columns]
    vif_df = pd.DataFrame(columns=["feature", "vif"])
    vif_method = "not_computed"
    if vif_features:
        vif_input = X_train[vif_features].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if not vif_input.empty:
            if STATSMODELS_AVAILABLE:
                vif_rows = []
                for index, feature in enumerate(vif_input.columns):
                    vif_rows.append(
                        {
                            "feature": feature,
                            "vif": round(
                                float(variance_inflation_factor(vif_input.values, index)),
                                4,
                            ),
                        }
                    )
                vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)
                vif_method = "statsmodels"
            else:
                vif_df = _compute_vif_fallback(vif_input)
                vif_method = "sklearn_fallback"

    model_selection_train = X_train_selection.copy().astype(float)
    model_selection_test = X_test_selection.copy().astype(float)
    binary_like_selection_features = [
        feature
        for feature in model_selection_train.columns
        if set(pd.Series(model_selection_train[feature]).dropna().unique()).issubset({0.0, 1.0})
    ]
    scale_for_model_features = [
        feature
        for feature in model_selection_train.columns
        if feature not in binary_like_selection_features
    ]
    scaler = StandardScaler()
    model_selection_train_scaled = model_selection_train.copy()
    model_selection_test_scaled = model_selection_test.copy()
    if scale_for_model_features:
        model_selection_train_scaled[scale_for_model_features] = scaler.fit_transform(
            model_selection_train_scaled[scale_for_model_features]
        )
        model_selection_test_scaled[scale_for_model_features] = scaler.transform(
            model_selection_test_scaled[scale_for_model_features]
        )

    l1_selector = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=random_state,
        max_iter=2000,
    )
    l1_selector.fit(model_selection_train_scaled, y_train)
    l1_selection_df = pd.DataFrame(
        {
            "feature": model_selection_train_scaled.columns,
            "l1_coefficient": np.round(l1_selector.coef_[0], 6),
            "l1_selected": np.abs(l1_selector.coef_[0]) > 1e-8,
        }
    )

    rfe_feature_count = max(10, min(20, model_selection_train_scaled.shape[1] // 3))
    rfe_selector = RFE(
        estimator=LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            max_iter=2000,
        ),
        n_features_to_select=rfe_feature_count,
    )
    rfe_selector.fit(model_selection_train_scaled, y_train)
    rfe_df = pd.DataFrame(
        {
            "feature": model_selection_train_scaled.columns,
            "rfe_selected": rfe_selector.support_,
            "rfe_rank": rfe_selector.ranking_,
        }
    )

    sfm_rf_selector = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    sfm_rf_selector.fit(model_selection_train, y_train)
    sfm_selector = SelectFromModel(sfm_rf_selector, threshold="median", prefit=True)
    select_from_model_df = pd.DataFrame(
        {
            "feature": model_selection_train.columns,
            "select_from_model_selected": sfm_selector.get_support(),
        }
    )

    selection_summary_df = pd.DataFrame({"feature": X_train_selection.columns})
    selection_summary_df["type"] = selection_summary_df["feature"].map(feature_type_map).fillna(
        "encoded_categorical_dummy"
    )
    selection_summary_df = selection_summary_df.merge(
        target_relationship_df,
        on="feature",
        how="left",
    )
    selection_summary_df = selection_summary_df.merge(mi_df, on="feature", how="left")
    selection_summary_df = selection_summary_df.merge(
        rf_importance_df,
        on="feature",
        how="left",
    )
    selection_summary_df = selection_summary_df.merge(
        l1_selection_df[["feature", "l1_selected"]],
        on="feature",
        how="left",
    )
    selection_summary_df = selection_summary_df.merge(
        rfe_df[["feature", "rfe_selected", "rfe_rank"]],
        on="feature",
        how="left",
    )
    selection_summary_df = selection_summary_df.merge(
        select_from_model_df,
        on="feature",
        how="left",
    )

    vif_map = dict(zip(vif_df["feature"], vif_df["vif"])) if not vif_df.empty else {}
    selection_summary_df["vif"] = selection_summary_df["feature"].map(vif_map)

    redundancy_note_map: dict[str, str] = {}
    for _, row in high_corr_pairs_df.iterrows():
        redundancy_note_map[row["feature_1"]] = (
            f"High correlation with {row['feature_2']} ({row['correlation']})"
        )
        redundancy_note_map[row["feature_2"]] = (
            f"High correlation with {row['feature_1']} ({row['correlation']})"
        )
    if "log_total_charges" in selection_summary_df["feature"].values:
        redundancy_note_map["log_total_charges"] = (
            "Cumulative-charge signal on the log scale; review alongside tenure, "
            "avg_monthly_spend, and tenure_x_MonthlyCharges."
        )
    if "Contract_ordinal" in selection_summary_df["feature"].values:
        redundancy_note_map["Contract_ordinal"] = (
            "Preferred primary contract representation for linear models; overlaps strongly "
            "with is_month_to_month."
        )
    if "is_month_to_month" in selection_summary_df["feature"].values:
        redundancy_note_map["is_month_to_month"] = (
            "High overlap with Contract_ordinal; keep as review/business flag rather "
            "than co-primary linear feature."
        )
    selection_summary_df["redundancy_note"] = (
        selection_summary_df["feature"].map(redundancy_note_map).fillna("")
    )
    selection_summary_df["selection_votes"] = (
        selection_summary_df["l1_selected"].fillna(False).astype(int)
        + selection_summary_df["rfe_selected"].fillna(False).astype(int)
        + selection_summary_df["select_from_model_selected"].fillna(False).astype(int)
    )
    selection_summary_df["final_decision"] = np.select(
        [
            selection_summary_df["selection_votes"] >= 2,
            (selection_summary_df["mutual_information"].fillna(0) >= 0.01)
            | (
                selection_summary_df["tree_importance"].fillna(0)
                >= selection_summary_df["tree_importance"].fillna(0).median()
            ),
        ],
        ["keep", "review"],
        default="drop",
    )
    selection_summary_df.loc[
        selection_summary_df["feature"].eq("log_total_charges"),
        "final_decision",
    ] = "review"
    selection_summary_df.loc[
        selection_summary_df["feature"].eq("is_month_to_month"),
        "final_decision",
    ] = "review"
    selection_summary_df.loc[
        selection_summary_df["feature"].eq("Contract_ordinal"),
        "final_decision",
    ] = "keep"
    selection_summary_df = selection_summary_df.sort_values(
        ["final_decision", "selection_votes", "mutual_information", "tree_importance"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    kept_features = selection_summary_df.loc[
        selection_summary_df["final_decision"] == "keep",
        "feature",
    ].tolist()
    review_features = selection_summary_df.loc[
        selection_summary_df["final_decision"] == "review",
        "feature",
    ].tolist()
    dropped_features = selection_summary_df.loc[
        selection_summary_df["final_decision"] == "drop",
        "feature",
    ].tolist()

    priority_review_features = {
        "is_month_to_month",
        "tenure",
        "MonthlyCharges",
        "log_total_charges",
        "tenure_x_MonthlyCharges",
        "avg_monthly_spend",
        "is_new_customer",
        "service_count",
    }
    priority_review_prefixes = (
        "contract_payment_profile_",
        "internet_techsupport_profile_",
        "internet_onlinesecurity_profile_",
    )
    candidate_final_features = [
        feature
        for feature in selection_summary_df["feature"]
        if (
            feature in kept_features
            or feature in priority_review_features
            or any(feature.startswith(prefix) for prefix in priority_review_prefixes)
        )
    ]

    manual_drop_features = set()
    duplicate_pairs = [
        ("internet_service_group_fiber", "InternetService_Fiber optic"),
        ("internet_service_group_dsl", "InternetService_DSL"),
        ("internet_service_group_no_internet", "InternetService_No"),
        ("is_new_customer", "tenure_band_new"),
        ("payment_method_group_electronic_check", "PaymentMethod_Electronic check"),
        ("payment_method_group_manual_check", "PaymentMethod_Mailed check"),
    ]
    for retained_feature, dropped_feature in duplicate_pairs:
        if {retained_feature, dropped_feature}.issubset(candidate_final_features):
            manual_drop_features.add(dropped_feature)

    selected_features = [
        feature for feature in candidate_final_features if feature not in manual_drop_features
    ]
    selected_features = list(dict.fromkeys(selected_features))

    full_feature_df = df.drop(columns=[target_column]).copy()
    full_object_features = full_feature_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    full_redundant_originals = [
        column for column in full_object_features if f"{column}_ordinal" in full_feature_df.columns
    ]
    full_encoded_categorical_features = [
        column for column in full_object_features if column not in full_redundant_originals
    ]
    full_X = pd.get_dummies(
        full_feature_df.drop(columns=full_redundant_originals, errors="ignore"),
        columns=full_encoded_categorical_features,
        drop_first=False,
        dtype=int,
    )

    missing_selected_features = [
        feature for feature in selected_features if feature not in full_X.columns
    ]
    if missing_selected_features:
        raise KeyError(
            f"Selected features missing from full feature matrix: {missing_selected_features}"
        )

    final_df = full_X[selected_features].copy()
    final_df[target_column] = df[target_column].astype(int)

    final_feature_decisions_df = pd.DataFrame(
        {
            "keep_features": pd.Series(kept_features, dtype="object"),
            "review_features": pd.Series(review_features, dtype="object"),
            "dropped_features": pd.Series(dropped_features, dtype="object"),
        }
    )

    dataset_summary = {
        "dataset_shape": df.shape,
        "target_column": target_column,
        "X_shape": X.shape,
        "y_shape": y.shape,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "test_size": test_size,
    }
    feature_type_summary = pd.DataFrame(
        {
            "feature_type": [
                "numerical",
                "binary",
                "encoded_categorical",
                "dropped_redundant_originals",
            ],
            "feature_count": [
                len(encoded["numerical_features"]),
                len(encoded["binary_features"]),
                len(encoded["encoded_categorical_features"]),
                len(encoded["redundant_originals"]),
            ],
            "features": [
                ", ".join(encoded["numerical_features"]),
                ", ".join(encoded["binary_features"]),
                ", ".join(encoded["encoded_categorical_features"]),
                ", ".join(encoded["redundant_originals"]),
            ],
        }
    )
    low_information_summary = {
        "constant_feature_count": len(encoded["constant_features"]),
        "near_constant_feature_count": len(encoded["near_constant_features"]),
        "near_constant_threshold": encoded["near_constant_threshold"],
        "remaining_feature_count": X_train_selection.shape[1],
    }
    final_selection_summary = {
        "features_before_selection": int(X.shape[1]),
        "features_after_selection": len(selected_features),
        "kept_feature_count": len(kept_features),
        "review_feature_count": len(review_features),
        "dropped_feature_count": len(dropped_features),
        "dropped_columns": dropped_features + sorted(manual_drop_features),
        "final_selected_features": selected_features,
    }

    return {
        "dataset_summary": dataset_summary,
        "feature_type_summary": feature_type_summary,
        "low_information_summary": low_information_summary,
        "target_relationship_df": target_relationship_df,
        "redundancy_review_df": high_corr_pairs_df,
        "mutual_information_df": mi_df,
        "tree_importance_df": rf_importance_df,
        "vif_df": vif_df,
        "vif_method": vif_method,
        "l1_selection_df": l1_selection_df,
        "rfe_df": rfe_df,
        "select_from_model_df": select_from_model_df,
        "selection_summary_df": selection_summary_df,
        "final_feature_decisions_df": final_feature_decisions_df,
        "final_selection_summary": final_selection_summary,
        "selected_features": selected_features,
        "final_df": final_df,
    }


def export_selected_dataset(
    results: dict[str, object],
    *,
    final_dataset_path: Path | None = None,
    feature_columns_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, object]:
    """Persist the final selected dataset, feature list, and core selection reports."""
    final_dataset_path = SELECTED_DATA_FILE if final_dataset_path is None else final_dataset_path
    feature_columns_path = FEATURE_COLUMNS_FILE if feature_columns_path is None else feature_columns_path
    report_dir = REPORTS_TABLES_DIR / "feature_selection" if report_dir is None else report_dir

    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    final_df = results["final_df"]
    selected_features = results["selected_features"]
    final_df.to_csv(final_dataset_path, index=False)
    with feature_columns_path.open("w", encoding="utf-8") as file_obj:
        json.dump(selected_features, file_obj, indent=2)

    pd.Series(selected_features, name="feature").to_csv(
        report_dir / "final_selected_feature_list.csv",
        index=False,
    )
    pd.DataFrame([results["dataset_summary"]]).to_csv(
        report_dir / "dataset_summary.csv",
        index=False,
    )
    pd.DataFrame([results["low_information_summary"]]).to_csv(
        report_dir / "low_information_summary.csv",
        index=False,
    )
    results["feature_type_summary"].to_csv(
        report_dir / "feature_type_summary.csv",
        index=False,
    )
    results["target_relationship_df"].to_csv(
        report_dir / "target_relationship_review.csv",
        index=False,
    )
    results["redundancy_review_df"].to_csv(
        report_dir / "redundancy_review.csv",
        index=False,
    )
    results["mutual_information_df"].to_csv(
        report_dir / "mutual_information_ranking.csv",
        index=False,
    )
    results["tree_importance_df"].to_csv(
        report_dir / "tree_feature_importance.csv",
        index=False,
    )
    results["vif_df"].to_csv(report_dir / "vif_review.csv", index=False)
    results["l1_selection_df"].to_csv(report_dir / "l1_selection.csv", index=False)
    results["rfe_df"].to_csv(report_dir / "rfe_selection.csv", index=False)
    results["select_from_model_df"].to_csv(
        report_dir / "select_from_model_selection.csv",
        index=False,
    )
    results["selection_summary_df"].to_csv(
        report_dir / "selection_summary.csv",
        index=False,
    )
    results["final_feature_decisions_df"].to_csv(
        report_dir / "final_feature_decisions.csv",
        index=False,
    )
    final_selection_summary = dict(results["final_selection_summary"])
    final_selection_summary["final_dataset_path"] = str(final_dataset_path.relative_to(PROJECT_ROOT))
    final_selection_summary["feature_columns_path"] = str(feature_columns_path.relative_to(PROJECT_ROOT))
    pd.DataFrame([final_selection_summary]).to_csv(
        report_dir / "final_selection_summary.csv",
        index=False,
    )

    return {
        "final_dataset_path": str(final_dataset_path.relative_to(PROJECT_ROOT)),
        "feature_columns_path": str(feature_columns_path.relative_to(PROJECT_ROOT)),
        "report_dir": str(report_dir.relative_to(PROJECT_ROOT)),
    }
