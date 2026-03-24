# Assignment 5 – MLOps CI/CD Pipeline

A fully-automated machine-learning pipeline using **GitHub Actions**, **MLflow**, **DVC**, and **Docker**.

## Architecture

```
push / PR
    │
    ▼
┌─────────────────────────── validate job ───────────────────────────┐
│  1. dvc pull          – restore fashion-mnist_train.csv            │
│  2. python train.py   – train RandomForest, log to MLflow          │
│  3. write model_info.txt  (Run ID)                                 │
│  4. upload model_info.txt as a workflow artefact                   │
└────────────────────────────────────────────────────────────────────┘
    │  needs: validate
    ▼
┌─────────────────────────── deploy job ─────────────────────────────┐
│  1. download model_info.txt artefact                               │
│  2. python check_threshold.py  – query MLflow, fail if acc < 0.85  │
│  3. docker build --build-arg RUN_ID=...  (mock build)              │
└────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml       ← GitHub Actions CI/CD
├── .dvc/
│   └── config                 ← DVC remote configuration
├── Dockerfile                 ← Container image definition
├── train.py                   ← Model training script
├── check_threshold.py         ← Accuracy gate (≥ 0.85)
├── requirements.txt
├── fashion-mnist_train.csv.dvc ← DVC pointer to dataset
└── README.md
```

## Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | URL of your MLflow tracking server (e.g. Dagshub) |
| `MLFLOW_TRACKING_USERNAME` | MLflow / Dagshub username |
| `MLFLOW_TRACKING_PASSWORD` | MLflow / Dagshub token or password |
| `DVC_GDRIVE_CREDENTIALS_DATA` | (Optional) Google Drive service-account JSON for DVC |

### Setting up Secrets

1. Go to **Settings → Secrets and variables → Actions** in your GitHub repo.
2. Click **New repository secret** and add each secret above.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install dvc

# Pull data (after configuring DVC remote)
dvc pull

# Train locally
python train.py

# Check threshold locally
python check_threshold.py
```

## Pipeline Gate

The `deploy` job will **fail** (blocking deployment) if the trained model's
accuracy is below **0.85**.  Adjust `ACCURACY_THRESHOLD` in `pipeline.yml` if needed.
