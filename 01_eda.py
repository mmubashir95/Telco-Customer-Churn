# =============================================================
# 📌 TELCO CUSTOMER CHURN - COMPLETE EDA + BASIC PIPELINE
# =============================================================
# This script covers:
# 1️⃣ Data understanding
# 2️⃣ Missing value detection
# 3️⃣ Outlier detection
# 4️⃣ Distribution analysis
# 5️⃣ Relationship analysis
# 6️⃣ Data leakage check
# 7️⃣ Preprocessing + Model pipeline
# =============================================================


# =============================================================
# 1️⃣ Import Required Libraries
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency


# =============================================================
# 2️⃣ Load Dataset
# =============================================================

# Load dataset (make sure path is correct)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =============================================================
# 🎯 GOAL 1: Understand Data Structure
# =============================================================

print("Dataset Shape:", df.shape)  # (rows, columns)

print("\nFirst 5 Rows:")
print(df.head())  # Preview data

print("\nDataset Info:")
df.info()  # Shows data types & memory usage

print("\nStatistical Summary:")
print(df.describe())  # Summary of numerical columns


# =============================================================
# 🎯 GOAL 2: Detect Missing Values
# =============================================================

print("\nMissing Values per Column:")
print(df.isna().sum())

# ⚠️ Important:
# TotalCharges is stored as string due to blank values.
# Convert it to numeric. Invalid values become NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges NaNs After Conversion:")
print(df["TotalCharges"].isna().sum())

# Inspect rows where TotalCharges became NaN
nan_rows = df[df["TotalCharges"].isna()]
print("\nRows with Missing TotalCharges:")
print(nan_rows[["tenure", "MonthlyCharges", "TotalCharges"]])


# =============================================================
# 🎯 GOAL 3: Detect Outliers
# =============================================================

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    # Skewness tells us distribution symmetry
    print(f"{col} Skew:", df[col].skew())


# =============================================================
# 🎯 GOAL 4: Understand Distributions
# =============================================================

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Target variable distribution
print("\nTarget Distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)


# =============================================================
# 🎯 GOAL 5: Check Relationships Between Variables
# =============================================================

# Convert target to numeric for correlation
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Correlation matrix (numeric columns only)
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation with Churn:")
print(corr_matrix["Churn"].sort_values(ascending=False))

# Scatter plot example
sns.scatterplot(x="tenure", y="TotalCharges", data=df)
plt.title("Tenure vs TotalCharges")
plt.show()


# =============================================================
# 🎯 GOAL 6: Detect Data Leakage
# =============================================================

print("\nChecking Potential Leakage Columns:")

for col in df.columns:
    if "churn" in col.lower() and col != "Churn":
        print("⚠️ Possible leakage column:", col)

# Manual inspection:
# - customerID → Identifier → Should be dropped
# - Any column created after churn event → Leakage risk


# =============================================================
# 🎯 GOAL 7: Decide Preprocessing Strategy
# =============================================================

print("\n--- Suggested Preprocessing Decisions ---")

# 1️⃣ Drop customerID (identifier)
df = df.drop("customerID", axis=1)

# 2️⃣ Handle TotalCharges missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("Remaining NaNs in TotalCharges:", df["TotalCharges"].isna().sum())

# 3️⃣ Separate Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4️⃣ Train-Test Split (VERY IMPORTANT before scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# 6️⃣ Preprocessing:
# - Scale numeric features
# - One-hot encode categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

# 7️⃣ Create Pipeline (Preprocessing + Model)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
])

# 8️⃣ Train Model
pipeline.fit(X_train, y_train)

print("\nModel Training Completed Successfully ✅")

# 2️⃣ Make predictions
y_pred = pipeline.predict(X_test)

# 3️⃣ Print evaluation
print(classification_report(y_test, y_pred))

# =============================================================
# 🚀 END OF SCRIPT
# =============================================================

# ==============================================
# CATEGORICAL FEATURE vs CHURN ANALYSIS
# ==============================================

# Ensure Churn is numeric (0/1)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove customerID if present
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

print("\n" + "="*80)
print("CATEGORICAL FEATURE ANALYSIS (Churn Rate per Category)")
print("="*80)

for col in categorical_cols:
    print("\n" + "="*60)
    print(f"Feature: {col}")
    print("="*60)

    churn_analysis = (
        df
        .groupby(col)["Churn"]
        .agg(["count", "mean"])
        .sort_values(by="mean", ascending=False)
    )

    churn_analysis.rename(columns={"mean": "churn_rate"}, inplace=True)
    churn_analysis["churn_rate_%"] = churn_analysis["churn_rate"] * 100

    print(churn_analysis)

print("\n" + "="*80)
print("CHI-SQUARE TEST (Categorical Features vs Churn)")
print("="*80)

# Ensure Churn is numeric (0/1)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove ID column if exists
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

for col in categorical_cols:
    print("\n" + "-"*60)
    print(f"Feature: {col}")
    print("-"*60)

    table = pd.crosstab(df[col], df["Churn"])

    chi2, p, dof, expected = chi2_contingency(table)

    print("Chi2 statistic:", chi2)
    print("Degrees of freedom:", dof)
    print("p-value:", p)