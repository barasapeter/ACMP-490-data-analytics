import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Q2 Dataset.csv")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing values:\n", df.isna().sum())

df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"], inplace=True)


def tenure_group(tenure):
    if tenure <= 6:
        return "0–6 Months"
    elif tenure <= 12:
        return "7–12 Months"
    elif tenure <= 24:
        return "13–24 Months"
    elif tenure <= 48:
        return "25–48 Months"
    else:
        return "49+ Months"


df["TenureGroup"] = df["tenure"].apply(tenure_group)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df["ChargePerMonth"] = df["TotalCharges"] / np.where(df["tenure"] == 0, 1, df["tenure"])

df["HasInternet"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)


X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=300, solver="lbfgs")),
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]
)

log_reg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)


def evaluate(model, model_name):
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n========== {model_name} ==========")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, pred_proba))

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


evaluate(log_reg_pipeline, "Logistic Regression")
evaluate(rf_pipeline, "Random Forest")


rf_model = rf_pipeline.named_steps["model"]
encoder = (
    rf_pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

encoded_cat_names = encoder.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, encoded_cat_names])

importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop Indicators of Churn:\n")
print(top_features)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind="barh")
plt.title("Top 15 Features Influencing Churn")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")
