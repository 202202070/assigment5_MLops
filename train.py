import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data   (DVC pulls bank.csv from DagsHub)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "bank.csv")

if os.path.exists(DATA_PATH):
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    # Synthetic fallback: 10-feature binary classification
    print(f"[INFO] {DATA_PATH} not found – generating synthetic bank-like data.")
    rng = np.random.default_rng(42)
    n = 11162
    df = pd.DataFrame(rng.integers(0, 100, (n, 10)),
                      columns=[f"f{i}" for i in range(10)])
    df["deposit"] = rng.choice(["yes", "no"], n)

print(f"Dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-process   (encode all categorical columns)
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "deposit"
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=[TARGET]).values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MLflow experiment
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
else:
    print("No MLFLOW_TRACKING_URI – using local ./mlruns")

mlflow.set_experiment("bank-deposit-classifier")

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 3))
MAX_DEPTH    = int(os.environ.get("MAX_DEPTH", 2))


with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    mlflow.log_param("n_estimators",  N_ESTIMATORS)
    mlflow.log_param("max_depth",     MAX_DEPTH)
    mlflow.log_param("dataset",       DATA_PATH)
    mlflow.log_param("train_samples", len(X_train))

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    # ── Export artifacts for the deploy job ──────────────────────────────────
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    with open("mlflow_uri.txt", "w") as f:
        f.write(mlflow.get_tracking_uri())

    print("model_info.txt  -> " + run_id)
    print("mlflow_uri.txt  -> " + mlflow.get_tracking_uri())


print("Training complete.")
