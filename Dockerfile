# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile  –  Fashion-MNIST Classifier Serving Image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Build-time argument: the MLflow run that produced the model
ARG RUN_ID
ENV MLFLOW_RUN_ID=${RUN_ID}

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy

# Copy inference helpers (if any)
COPY . /app

# Simulate downloading the model artefact from MLflow
# (In production replace this with:  mlflow artifacts download --run-id ${MLFLOW_RUN_ID} -a model -d /app/model )
RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    mkdir -p /app/model && \
    echo "Model artefact placeholder for run ${RUN_ID}" > /app/model/README.txt

EXPOSE 8080

# Default command – replace with your actual serving command
CMD ["python", "-c", "print('Model server started for Run ID:', '${MLFLOW_RUN_ID}')"]
