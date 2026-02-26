# =============================================================
# 02_train_pipeline.py
# TELCO CHURN - COMPLETE ML PIPELINE (CLEAN -> PREPROCESS -> TRAIN -> EVAL -> SAVE)
# Uses your same core techniques from 01_eda.py but in a proper sklearn pipeline.
# =============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import joblib


# -----------------------------
# 1) Load + Clean Data (your techniques)
# -----------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop identifier (your note: customerID should be dropped)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges (your exact technique)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Your logic idea: tenure==0 often causes blanks -> set TotalCharges to 0
        if "tenure" in df.columns:
            df.loc[(df["tenure"] == 0) & (df["TotalCharges"].isna()), "TotalCharges"] = 0

    # Convert target to 0/1 (your technique)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# -----------------------------
# 2) Build Preprocessor
# -----------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown avoids crash if new category appears in test/real data
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


# -----------------------------
# 3) Train + Evaluate
# -----------------------------
def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # predict_proba exists for these models; safe check anyway
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    print("\n" + "=" * 80)
    print(f"MODEL: {name}")
    print("=" * 80)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print("ROC-AUC:", round(auc, 4))
        return auc

    return None


def main():
    # ✅ IMPORTANT: set your correct path
    # If you have it in project: data/WA_Fn-UseC_-Telco-Customer-Churn.csv keep that.
    # Otherwise use the absolute path.
    CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {CSV_PATH}\n"
            f"Fix the path. Example:\n"
            f'CSV_PATH = "/Users/mohammadmubashir/VCode/Telco-Customer-Churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"'
        )

    df = load_and_clean(CSV_PATH)

    # -----------------------------
    # Split features/target
    # -----------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Stratify preserves churn ratio in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # -----------------------------
    # Models (compare)
    # -----------------------------
    models = {
        # Your current approach: LogisticRegression with class_weight balanced
        "LogReg (class_weight=balanced)": LogisticRegression(
            class_weight="balanced", max_iter=2000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_pipe, best_auc = None, None, -1

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        auc = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        # If AUC is None (rare here), fallback to -1
        auc = auc if auc is not None else -1

        # Track best by AUC (you can change to F1 for churn if you want)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipe = pipe

    print("\n" + "#" * 80)
    print("BEST MODEL:", best_name)
    print("BEST ROC-AUC:", round(best_auc, 4))
    print("#" * 80)

    # -----------------------------
    # Optional: Cross-validation on BEST model (more reliable)
    # -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print("\nCV ROC-AUC scores:", np.round(cv_scores, 4))
    print("CV Mean ROC-AUC:", round(cv_scores.mean(), 4))

    # -----------------------------
    # Save best pipeline
    # -----------------------------
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/telco_churn_best_pipeline.joblib"
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, out_path)
    print("\n✅ Saved best pipeline to:", out_path)

    # -----------------------------
    # Example: Load + predict on new data
    # -----------------------------
    loaded = joblib.load(out_path)
    sample_pred = loaded.predict(X_test.iloc[:5])
    sample_prob = loaded.predict_proba(X_test.iloc[:5])[:, 1]
    print("\nSample predictions (first 5):", sample_pred)
    print("Sample churn probabilities (first 5):", np.round(sample_prob, 4))

    # Create new customer manually
    new_customer = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }])

    # Predict
    prediction = loaded.predict(new_customer)
    probability = loaded.predict_proba(new_customer)[:, 1]

    print("Churn Prediction:", prediction[0])
    print("Churn Probability:", probability[0])


if __name__ == "__main__":
    main()