"""
check_threshold.py
──────────────────
Reads the MLflow Run ID from model_info.txt, fetches the logged
accuracy metric, and exits with a non-zero status if the accuracy
is below the required threshold.
"""

import sys
import os
import mlflow

THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", 0.85))
MODEL_INFO_FILE = os.environ.get("MODEL_INFO_FILE", "model_info.txt")

# ── Read Run ID ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_INFO_FILE):
    print(f"[ERROR] {MODEL_INFO_FILE} not found. Did the validate job succeed?")
    sys.exit(1)

with open(MODEL_INFO_FILE, "r") as f:
    run_id = f.read().strip()

if not run_id:
    print("[ERROR] model_info.txt is empty.")
    sys.exit(1)

print(f"Checking Run ID: {run_id}")

# ── Connect to MLflow ──────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.tracking.MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"[ERROR] Could not retrieve run {run_id} from MLflow: {e}")
    sys.exit(1)

accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print(f"[ERROR] 'accuracy' metric not found in run {run_id}.")
    sys.exit(1)

print(f"Accuracy from MLflow: {accuracy:.4f}")
print(f"Required threshold  : {THRESHOLD}")

if accuracy < THRESHOLD:
    print(
        f"[FAIL] Accuracy {accuracy:.4f} is BELOW the threshold {THRESHOLD}. "
        "Deployment blocked."
    )
    sys.exit(1)

print(
    f"[PASS] Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}. "
    "Proceeding to deployment."
)
sys.exit(0)
