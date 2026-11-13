import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/isolation_forest_model.joblib"
model = None


# Manage the application lifecycle with FastAPI's lifespan handler.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Load the model during application startup.
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[+] Model loaded successfully from {MODEL_PATH}")
    else:
        print(
            f"[!] Warning: Model file not found at {MODEL_PATH}. "
            "Train the model first."
        )
    yield

    # Run cleanup tasks during application shutdown, if needed.
    print("[*] Shutting down SecureAI-Guard API...")


# Initialize the FastAPI application with the lifespan handler.
app = FastAPI(
    title="SecureAI-Guard Inference API",
    description="Real-time anomaly detection API (Isolation Forest)",
    version="1.0.0",
    lifespan=lifespan,
)


# Define the expected format of the input JSON payload.
class SecurityLogInput(BaseModel):
    failed_logins: float
    request_rate: float
    payload_size: float
    session_duration: float


@app.get("/")
def read_root():
    return {"status": "online", "service": "SecureAI-Guard ML Engine"}


@app.post("/predict")
def predict_anomaly(log: SecurityLogInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded on the server.",
        )

    input_data = pd.DataFrame(
        [
            {
                "failed_logins": log.failed_logins,
                "request_rate": log.request_rate,
                "payload_size": log.payload_size,
                "session_duration": log.session_duration,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    anomaly_score = model.decision_function(input_data)[0]
    is_anomaly = bool(prediction == -1)

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": float(anomaly_score),
        "status": "THREAT_DETECTED" if is_anomaly else "NORMAL",
    }
