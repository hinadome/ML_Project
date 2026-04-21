import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import MinMaxScaler
from contextlib import asynccontextmanager

COUNT_PER_INSTANCE=1000

# --- 1. CONFIGURATION & MODEL LOADING ---
# In a real scenario, these paths would point to your saved .joblib or .pth files
MODELS = {
    "gbr": "./model/best_gbr.pkl",
    "xgb": "./model/best_xgb.pkl",
    "anomaly": "./model/anomaly_model.pkl"
}

# Placeholder for loaded models
loaded_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models into memory when the service starts.
    """
    try:
        loaded_models["gbr"] = joblib.load(MODELS["gbr"])
        loaded_models["xgb"] = joblib.load(MODELS["xgb"])
        loaded_models["anomaly"] = joblib.load(MODELS["anomaly"])
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}. Ensure artifacts exist.")
    yield
    # Shutdown code if needed

app = FastAPI(
    title="Cloud Operations ML Service",
    description="API for Proactive Auto-Scaling and Log Anomaly Detection",
    version="0.1.0",
    lifespan=lifespan
)

# --- 2. SCHEMAS ---
class LogEntry(BaseModel):
    request_count: int
    error_5xx: int
    bytes_sum: int
    hour: int

class PredictionRequest(BaseModel):
    history: List[LogEntry] 

def build_tree_features(df):
    x = df.copy().sort_values(by="hour")
    #for lag in [1, 2, 24]:
    for lag in [1, 2]:
        x[f"lag_{lag}"] = x["request_count"].shift(lag)
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24)
    x = x.dropna()
    return x.drop('hour', axis=1)

# --- 3. ENDPOINTS ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": len(loaded_models) > 0}

@app.post("/predict-scaling")
async def predict_scaling(data: PredictionRequest):
    """
    Receives recent logs and returns the predicted traffic for the next hour.
    """
    if "xgb" not in loaded_models:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    if len(data.history) < 24:
        raise HTTPException(status_code=400, detail="Need at least 24 hours of history for prediction.")
    
    # Convert input to DataFrame
    df = pd.DataFrame([item.model_dump() for item in data.history])

    features = build_tree_features(df)

    prediction = loaded_models["xgb"].predict(features)
    predicted_request_count = prediction.mean() * 1.1 
    return {
        "model": "xgb",
        "predicted_request_count": float(predicted_request_count),
        "recommended_instances": int(np.ceil(predicted_request_count / COUNT_PER_INSTANCE)) # e.g., 100 reqs per instance
    }

@app.post("/predict-scaling-with-gbr")
async def predict_scaling(data: PredictionRequest):
    """
    Receives recent logs and returns the predicted traffic for the next hour.
    """
    if "xgb" not in loaded_models:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    if len(data.history) < 24:
        raise HTTPException(status_code=400, detail="Need at least 24 hours of history for prediction.")
    
    # Convert input to DataFrame
    df = pd.DataFrame([item.model_dump() for item in data.history])

    features = build_tree_features(df)

    prediction = loaded_models["gbr"].predict(features)
    predicted_request_count = prediction.mean() * 1.1 
    return {
        "model": "gbr",
        "predicted_request_count": float(predicted_request_count),
        "recommended_instances": int(np.ceil(predicted_request_count / COUNT_PER_INSTANCE)) # e.g., 100 reqs per instance
    }

@app.post("/detect-anomalies")
async def detect_anomalies(data: List[LogEntry]):
    """
    Analyzes a batch of logs and returns indices of detected anomalies.
    """
    if "anomaly" not in loaded_models:
        raise HTTPException(status_code=500, detail="Anomaly model not loaded. Check server logs.")
    
    df = pd.DataFrame([item.model_dump() for item in data])

    # Features for Isolation Forest
    features = df[['request_count', 'error_5xx', 'bytes_sum']]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)
    
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