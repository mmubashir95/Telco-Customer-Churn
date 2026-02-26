import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1) Load data
# =========================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =========================
# 2) Basic cleaning
# =========================
# Convert TotalCharges to numeric (some rows are blank -> become NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop customerID (identifier, not useful for prediction)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Drop rows where TotalCharges became NaN (usually very few, often new customers)
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

# Target: Churn Yes/No -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 3) Split X / y
# =========================
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify numeric/categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

# =========================
# 4) Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# =========================
# 5) Full Pipeline (Preprocess + Model)
# =========================
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ]
)

# =========================
# 6) Train-test split (IMPORTANT: split BEFORE fitting pipeline)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # keeps churn ratio similar in train/test
)

# =========================
# 7) Train
# =========================
model.fit(X_train, y_train)

# =========================
# 8) Evaluate
# =========================
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))