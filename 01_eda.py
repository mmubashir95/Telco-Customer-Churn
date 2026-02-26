# =============================================================
# 📌 TELCO CUSTOMER CHURN - COMPLETE EDA + BASIC PIPELINE
# =============================================================
# This script covers (in order):
# 1️⃣ Data understanding
# 2️⃣ Missing value detection
# 3️⃣ Outlier detection
# 4️⃣ Distribution analysis + target balance
# 5️⃣ Relationship analysis (correlation, scatter)
# 6️⃣ Data leakage checks (quick heuristics)
# 7️⃣ Preprocessing decisions
# 8️⃣ ML pipeline (preprocess + Logistic Regression)
# 9️⃣ Categorical feature analysis + Chi-square tests
#
# ✅ NOTE:
# - I did NOT change your logic/steps (same operations).
# - I only formatted + re-ordered sections for readability and added comments.
# =============================================================


# =============================================================
# 1️⃣ Import Required Libraries
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency


# =============================================================
# ✅ Helper: Pretty Missing-Value Percentage Report
# =============================================================

def print_missing_report(df: pd.DataFrame, title: str = "Missing Value Percentage Report") -> None:
    """Print missing-value % per column (sorted), showing only columns with missing values."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)

    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_percent = missing_percent[missing_percent > 0]

    if missing_percent.empty:
        print("✅ No missing values found in dataset.")
    else:
        print(missing_percent)

    print("=" * 60 + "\n")


# =============================================================
# 2️⃣ Load Dataset
# =============================================================

# Load dataset (make sure path is correct for your project)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Missing % check (RAW dataset)
print_missing_report(df, title="Missing % before cleaning")


# =============================================================
# 🎯 GOAL 1: Understand Data Structure
# =============================================================

print("Dataset Shape:", df.shape)  # (rows, columns)

print("\nFirst 5 Rows:")
print(df.head())  # Preview the first 5 rows

print("\nDataset Info:")
df.info()  # Data types, non-null counts, memory usage

print("\nStatistical Summary:")
print(df.describe())  # Summary stats of numerical columns


# =============================================================
# 🎯 GOAL 2: Detect Missing Values
# =============================================================

print("\nMissing Values per Column:")
print(df.isna().sum())

# ⚠️ Important:
# - TotalCharges is often stored as a string because of blank values (" ").
# - Converting with errors="coerce" turns those invalid/blanks into NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Missing % check (after converting TotalCharges)
print_missing_report(df, title="Missing % after converting TotalCharges")

print("\nTotalCharges NaNs After Conversion:")
print(df["TotalCharges"].isna().sum())

# Inspect rows where TotalCharges became NaN
nan_rows = df[df["TotalCharges"].isna()]
print("\nRows with Missing TotalCharges:")
print(nan_rows[["tenure", "MonthlyCharges", "TotalCharges"]])


# =============================================================
# 🎯 GOAL 3: Detect Outliers (Boxplot) + Skewness
# =============================================================

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_cols:
    # Boxplot helps visualize outliers using quartiles (IQR)
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    # Skewness tells distribution symmetry:
    # - positive skew => right tail
    # - negative skew => left tail
    print(f"{col} Skew:", df[col].skew())

# Loop through each numerical column to detect outliers using IQR method
for col in numeric_cols:

    # Calculate 1st Quartile (25th percentile)
    Q1 = df[col].quantile(0.25)

    # Calculate 3rd Quartile (75th percentile)
    Q3 = df[col].quantile(0.75)

    # Compute Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define lower boundary for outlier detection
    # Any value below this will be considered an outlier
    lower = Q1 - 1.5 * IQR

    # Define upper boundary for outlier detection
    # Any value above this will be considered an outlier
    upper = Q3 + 1.5 * IQR

    # Identify rows where values fall outside the lower and upper bounds
    # These rows are potential outliers
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    # Calculate and print percentage of outliers in this column
    # (number of outlier rows divided by total dataset size)
    print(col, "Outlier %:", len(outliers)/len(df)*100)

# =============================================================
# 🎯 GOAL 4: Understand Distributions + Target Balance
# =============================================================

for col in numeric_cols:
    # Histogram + KDE curve to understand distribution shape
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Target variable distribution (before converting to numeric)
print("\nTarget Distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)


# =============================================================
# 🎯 GOAL 5: Check Relationships Between Variables
# =============================================================

# Convert target to numeric for correlation (Yes → 1, No → 0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Correlation matrix (numeric columns only)
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation with Churn:")
print(corr_matrix["Churn"].sort_values(ascending=False))

# Scatter plot example: tenure vs TotalCharges
sns.scatterplot(x="tenure", y="TotalCharges", data=df)
plt.title("Tenure vs TotalCharges")
plt.show()


# =============================================================
# 🎯 GOAL 6: Detect Data Leakage (quick check)
# =============================================================

print("\nChecking Potential Leakage Columns:")

# Heuristic check: columns that contain word 'churn' besides the target itself
for col in df.columns:
    if "churn" in col.lower() and col != "Churn":
        print("⚠️ Possible leakage column:", col)

# Manual inspection reminders:
# - customerID → Identifier → should be dropped
# - Any feature created AFTER churn event → leakage risk


# =============================================================
# 🎯 GOAL 7: Decide Preprocessing Strategy
# =============================================================

print("\n--- Suggested Preprocessing Decisions ---")

# 1️⃣ Drop customerID (identifier column)
df = df.drop("customerID", axis=1)

# 2️⃣ Handle TotalCharges missing values (from earlier conversion)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

median_total = df.loc[df["tenure"] > 0, "TotalCharges"].median()

df["TotalCharges"] = np.where(
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    0,
    df["TotalCharges"]
)

df["TotalCharges"].fillna(median_total, inplace=True)

print("Remaining NaNs in TotalCharges:", df["TotalCharges"].isna().sum())

# Missing % check (after handling missing values)
print_missing_report(df, title="Missing % after handling missing values")

# 3️⃣ Separate Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4️⃣ Train-Test Split (VERY IMPORTANT: always split before scaling/encoding!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# 6️⃣ Preprocessing:
# - Scale numeric features (StandardScaler)
# - One-hot encode categorical features (OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
    ]
)

# 7️⃣ Create Pipeline (Preprocessing + Model)
pipeline = Pipeline(
    [
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
)

# 8️⃣ Train Model
pipeline.fit(X_train, y_train)

print("\nModel Training Completed Successfully ✅")

# 9️⃣ Make predictions
y_pred = pipeline.predict(X_test)

# 🔟 Print evaluation metrics
print(classification_report(y_test, y_pred))


# =============================================================
# 8️⃣ CATEGORICAL FEATURE vs CHURN ANALYSIS
# =============================================================
# This section helps you see churn rate per category (e.g., Contract type).
# Useful for feature understanding + business insights.

# Ensure Churn is numeric (0/1)
# (This is defensive: if you run this block alone in isolation, it still works.)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns (object type)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove customerID if present (defensive; you already dropped it earlier)
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

print("\n" + "=" * 80)
print("CATEGORICAL FEATURE ANALYSIS (Churn Rate per Category)")
print("=" * 80)

for col in categorical_cols:
    print("\n" + "=" * 60)
    print(f"Feature: {col}")
    print("=" * 60)

    # For each category:
    # - count = number of records in that category
    # - mean = churn rate because Churn is 0/1 (mean of 0/1 = proportion of 1s)
    churn_analysis = (
        df.groupby(col)["Churn"]
        .agg(["count", "mean"])
        .sort_values(by="mean", ascending=False)
    )

    churn_analysis.rename(columns={"mean": "churn_rate"}, inplace=True)
    churn_analysis["churn_rate_%"] = churn_analysis["churn_rate"] * 100

    print(churn_analysis)


# =============================================================
# 9️⃣ CHI-SQUARE TEST (Categorical Features vs Churn)
# =============================================================
# Chi-square checks whether churn is statistically associated with a category.
# Rule of thumb:
# - p < 0.05  => significant relationship (likely useful feature)
# - p >= 0.05 => not significant (may still help in ML, but weaker evidence)

print("\n" + "=" * 80)
print("CHI-SQUARE TEST (Categorical Features vs Churn)")
print("=" * 80)

# Ensure Churn is numeric (0/1) (defensive again)
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Identify categorical columns again (kept as-is; useful if you run blocks separately)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove ID column if exists
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

for col in categorical_cols:
    print("\n" + "-" * 60)
    print(f"Feature: {col}")
    print("-" * 60)

    # Contingency table: category values x churn(0/1)
    table = pd.crosstab(df[col], df["Churn"])

    chi2, p, dof, expected = chi2_contingency(table)

    print("Chi2 statistic:", chi2)
    print("Degrees of freedom:", dof)
    print("p-value:", p)


# =============================================================
# Final Missing-Value Check (Clean Output)
# =============================================================

print_missing_report(df, title="Final missing % (end of script)")

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    print("\n" + "="*60)
    print(f"{col.upper()} DISTRIBUTION (%)")
    print("="*60)
    print((df[col].value_counts(normalize=True) * 100).round(2))
    print("="*60)


# ============================================================
# CHURN RATE PER CATEGORICAL FEATURE
# ============================================================

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    print("\n" + "="*60)
    print(f"CHURN DISTRIBUTION FOR: {col.upper()}")
    print("="*60)
    
    churn_table = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    print(churn_table.round(2))
    
    print("="*60)


# =============================================================
# 🚀 END OF SCRIPT
# =============================================================