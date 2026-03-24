"""
check_threshold.py
──────────────────
Reads the MLflow Run ID from model_info.txt, fetches the logged
accuracy metric, and exits with a non-zero status if the accuracy
is below the required threshold.

When no remote MLflow server is configured, falls back to the local
mlruns directory (written by train.py in the same job context).
"""

import sys
import os
import mlflow

THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", 0.85))
MODEL_INFO_FILE = os.environ.get("MODEL_INFO_FILE", "model_info.txt")
MLFLOW_URI_FILE = "mlflow_uri.txt"

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
# Priority: env var (remote server) > mlruns/ artifact (local file tracking)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

if not MLFLOW_TRACKING_URI:
    # Read what train.py wrote, but if it's a machine-specific SQLite path,
    # replace it with the relative ./mlruns directory that was downloaded as an artifact.
    if os.path.exists(MLFLOW_URI_FILE):
        with open(MLFLOW_URI_FILE, "r") as f:
            saved_uri = f.read().strip()
        if saved_uri.startswith("sqlite:") or saved_uri.startswith("/"):
            # Absolute path from a different runner – use the downloaded mlruns/ instead
            MLFLOW_TRACKING_URI = "./mlruns"
            print(f"[INFO] Saved URI was machine-specific ({saved_uri}), using ./mlruns artifact instead.")
        else:
            MLFLOW_TRACKING_URI = saved_uri
            print(f"[INFO] Using tracking URI from {MLFLOW_URI_FILE}: {MLFLOW_TRACKING_URI}")
    else:
        MLFLOW_TRACKING_URI = "./mlruns"
        print("[INFO] No mlflow_uri.txt found – defaulting to ./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"[INFO] MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")


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

print(f"Accuracy from MLflow : {accuracy:.4f}")
print(f"Required threshold   : {THRESHOLD}")

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
