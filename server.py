import torch
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Cloud Operations ML Service",
    description="API for Proactive Auto-Scaling and Log Anomaly Detection",
    version="0.1.0"
)

# --- 1. CONFIGURATION & MODEL LOADING ---
# In a real scenario, these paths would point to your saved .joblib or .pth files
MODELS = {
    "gbr": "./model/GBR_model.pkl",
    "anomaly": "./model/anomaly_model.pkl",
}

# Placeholder for loaded models
loaded_models = {}

@app.on_event("startup")
async def load_models():
    """
    Load models into memory when the service starts.
    """
    try:
        loaded_models["gbr"] = joblib.load(MODELS["gbr"])
        loaded_models["anomaly"] = joblib.load(MODELS["anomaly"])
        
        print("Models loaded successfully (Placeholder logic implemented)")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure artifacts exist.")

# --- 2. SCHEMAS ---
class LogEntry(BaseModel):
    request_count: int
    error_5xx: int
    bytes_sum: int
    hour: int

class PredictionRequest(BaseModel):
    history: List[LogEntry] 


def build_tree_features(data):
    return data

# --- 3. ENDPOINTS ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": len(loaded_models) > 0}

@app.post("/predict-scaling")
async def predict_scaling(data: PredictionRequest):
    """
    Receives recent logs and returns the predicted traffic for the next hour.
    """
    if len(data.history) < 24:
        raise HTTPException(status_code=400, detail="Need at least 24 hours of history for prediction.")
    
    # Convert input to DataFrame
    df = pd.DataFrame([item.model_dump() for item in data.history])
    
    # Example logic for XGBoost prediction
    # 1. Feature Engineering (similar to build_tree_features)
    # 2. prediction = loaded_models["xgb"].predict(features)
    features = build_tree_features(df)
    prediction = load_models["gbr"].predict(features)
    
    return {
        "model": "gbr",
        "predicted_request_count": float(prediction),
        "recommended_instances": int(np.ceil(prediction / 100)) # e.g., 100 reqs per instance
    }

@app.post("/detect-anomalies")
async def detect_anomalies(data: List[LogEntry]):
    """
    Analyzes a batch of logs and returns indices of detected anomalies.
    """
    df = pd.DataFrame([item.model_dump() for item in data])
    
    # Features for Isolation Forest
    features = df[['request_count', 'error_5xx', 'bytes_sum']]
    
    score = loaded_models["anomaly"].predict(features)
    is_anomaly = 1 if score == -1 else 0
    # -1 is anomaly, 1 is normal
    
    return {
        "total_processed": len(df),
        "anomalies_detected": is_anomaly, # Placeholder
        "indices": []
    }

# --- 4. RUNNING THE SERVICE ---
if __name__ == "__main__":
    import uvicorn
    # In cloud, you'd run: uvicorn app:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host="127.0.0.1", port=8000)