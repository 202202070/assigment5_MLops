"""
check_threshold.py
──────────────────
Reads model_info.txt (produced by train.py in the validate job):
  Line 1 – MLflow Run ID
  Line 2 – accuracy score

Fails the pipeline (exit 1) if accuracy < ACCURACY_THRESHOLD.
No MLflow server connection needed in the deploy job.
"""

import sys
import os

THRESHOLD         = float(os.environ.get("ACCURACY_THRESHOLD", 0.85))
MODEL_INFO_FILE   = os.environ.get("MODEL_INFO_FILE", "model_info.txt")

# ── Read model_info.txt ────────────────────────────────────────────────────────
if not os.path.exists(MODEL_INFO_FILE):
    print(f"[ERROR] {MODEL_INFO_FILE} not found. Did the validate job succeed?")
    sys.exit(1)

with open(MODEL_INFO_FILE, "r") as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

if len(lines) < 2:
    print(f"[ERROR] {MODEL_INFO_FILE} must have 2 lines: run_id and accuracy.")
    print(f"        Found: {lines}")
    sys.exit(1)

run_id   = lines[0]
accuracy = float(lines[1])

print(f"Run ID           : {run_id}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Required threshold: {THRESHOLD}")

# ── Threshold gate ────────────────────────────────────────────────────────────
if accuracy < THRESHOLD:
    print(
        f"\n[FAIL] Accuracy {accuracy:.4f} is BELOW the threshold {THRESHOLD}.\n"
        "       Deployment blocked."
    )
    sys.exit(1)

print(
    f"\n[PASS] Accuracy {accuracy:.4f} meets the threshold {THRESHOLD}.\n"
    "       Proceeding to deployment."
)
sys.exit(0)
