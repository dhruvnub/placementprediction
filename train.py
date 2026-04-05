# train.py (MLflow clean version)

import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import joblib

# ─────────────────────────────────────────────
# MLflow Setup (FIXED)
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("placement-prediction")

MODEL_NAME = "placement-classifier"

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = pd.read_csv("data/placementdata.csv")

FEATURES = [
    "CGPA", "Internships", "Projects",
    "AptitudeTestScore", "SoftSkillsRating",
    "SSC_Marks", "HSC_Marks"
]

X = df[FEATURES]
y = (df["PlacementStatus"] == "Placed").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=150, max_depth=8,
        min_samples_split=5, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=500, random_state=42
    )
}

best_f1 = 0
best_model = None
best_name = None

# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
for model_name, clf in candidates.items():
    with mlflow.start_run(run_name=model_name):

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1").mean()

        # Log params + metrics
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_f1", cv_f1)

        # Log model (FIXED)
        mlflow.sklearn.log_model(
            clf,
            name="model",
            registered_model_name=MODEL_NAME
        )

        print(f"\n[{model_name}]")
        for k, v in metrics.items():
            print(f"  {k}: {round(v,4)}")

        # Best model selection
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = clf
            best_name = model_name

# ─────────────────────────────────────────────
# Save best model locally
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/model.pkl")

with open("models/metadata.json", "w") as f:
    json.dump({
        "model_type": best_name,
        "f1": best_f1
    }, f, indent=2)

print("\nBest Model:", best_name)
print("F1 Score :", best_f1)
print("\nMLflow UI → mlflow ui")