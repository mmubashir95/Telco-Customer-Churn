# =============================================================
# 🎯 1️⃣ Import Libraries
# =============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# 🎯 2️⃣ Load Dataset
# =============================================================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =============================================================
# 🎯 GOAL 1: Understand Data Structure
# =============================================================

print("Dataset Shape:", df.shape)              # Rows & columns
print("\nFirst 5 Rows:\n", df.head())         # Preview data
print("\nDataset Info:\n")
df.info()                                      # Data types & memory
print("\nStatistical Summary:\n", df.describe())


# =============================================================
# 🎯 GOAL 2: Detect Missing Values
# =============================================================

print("\nMissing Values per Column:\n", df.isna().sum())

# Convert TotalCharges to numeric (important!)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTotalCharges NaNs After Conversion:",
      df["TotalCharges"].isna().sum())


# =============================================================
# 🎯 GOAL 3: Detect Outliers
# =============================================================

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

    print(f"{col} Skew:", df[col].skew())


# =============================================================
# 🎯 GOAL 4: Understand Distributions
# =============================================================

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Target Distribution
print("\nTarget Distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)


# =============================================================
# 🎯 GOAL 5: Check Relationships Between Variables
# =============================================================

# Encode target for correlation analysis
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Correlation Matrix (numeric only)
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Correlation with Target
print("\nCorrelation with Churn:")
print(corr_matrix["Churn"].sort_values(ascending=False))

# Scatter Example
sns.scatterplot(x="tenure", y="TotalCharges", data=df)
plt.title("Tenure vs TotalCharges")
plt.show()


# =============================================================
# 🎯 GOAL 6: Detect Data Leakage
# =============================================================

# Check if any column directly reveals target
print("\nChecking Potential Leakage Columns:")

for col in df.columns:
    if "churn" in col.lower() and col != "Churn":
        print("⚠️ Possible leakage column:", col)

# Also manually inspect columns like:
# customerID (identifier → should drop)
# Any column created after churn event


# =============================================================
# 🎯 GOAL 7: Decide Preprocessing Strategy
# =============================================================

print("\n--- Suggested Preprocessing Decisions ---")

print("1. Drop customerID (identifier).")
print("2. Handle TotalCharges missing values.")
print("3. Encode categorical features (One-Hot).")
print("4. Scale numerical features (StandardScaler).")
print("5. Handle class imbalance if needed.")