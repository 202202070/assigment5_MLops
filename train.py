import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# 1. Load data (DVC-tracked CSV)
# ──────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "fashion-mnist_train.csv")

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"]).values
y = df["label"].values

# Use a subset for speed in CI
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 10000))
if len(X) > SAMPLE_SIZE:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), SAMPLE_SIZE, replace=False)
    X, y = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 2. MLflow experiment
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
else:
    # Fall back to local file-based tracking
    print("No MLFLOW_TRACKING_URI set – using local ./mlruns")

mlflow.set_experiment("fashion-mnist-classifier")

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH    = int(os.environ.get("MAX_DEPTH", 15))

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Log hyper-parameters
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("sample_size", SAMPLE_SIZE)

    # Train
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")

    # ──────────────────────────────────────────
    # 3. Export metadata so the deploy job can read it
    # ──────────────────────────────────────────
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    # Also write the tracking URI so check_threshold.py can use it
    # even when running in a different job (it reads from model_info.txt
    # but needs the same URI).
    with open("mlflow_uri.txt", "w") as f:
        f.write(mlflow.get_tracking_uri())

    print(f"Run ID written to model_info.txt : {run_id}")
    print(f"Tracking URI written to mlflow_uri.txt: {mlflow.get_tracking_uri()}")

print("Training complete.")
