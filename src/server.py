import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from typing import List
from .schema import LogInstance, ScalingRequest, ScalingResponse, SmartScalingResponse, AnomalyDetectionResponse, HealthResponse

COUNT_PER_INSTANCE=100000
NORMAL_ADJUST_COUNT=10000
ABNORMAL_ADJUST_COUNT=25000

app = FastAPI(title="Proactive Traffic Scaler & Anomaly Detector")

# --- LOAD ARTIFACTS ---
try:
    xgb_model = joblib.load("model/best_xgb.pkl")
    gbr_model = joblib.load("model/best_gbr.pkl")
    anomaly_model = joblib.load("model/anomaly_model.pkl")
    scaler_x = joblib.load("model/scaler_x.pkl")
except Exception as e:
    print(f"Warning: Model artifacts not fully loaded: {e}")

def engineer_features(history: List[LogInstance]):
    df = pd.DataFrame([inst.model_dump() for inst in history]).reset_index(drop=True)
    df['lag_24h'] = df['request_count'].shift(24)
    df['lag_1h'] = df['request_count'].shift(1)
    df['lag_2h'] = df['request_count'].shift(2)
    df['rolling_mean_3h'] = df['request_count'].rolling(window=3).mean()
    df['velocity'] = df['request_count'].diff() / (df['request_count'].shift(1) + 1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    feature_cols = [
        'request_count', 'error_5xx', 'bytes_sum', 
        'lag_24h', 'lag_1h', 'lag_2h', 
        'rolling_mean_3h', 'velocity', 
        'hour_sin', 'hour_cos'
    ]
    latest_row = df.tail(1).copy()
    inference_data = latest_row[feature_cols]
    if inference_data.isnull().any().any():
        raise ValueError("Insufficient history. Provide 25+ hours of data.")
    return inference_data

def check_anomaly_internal(history: List[LogInstance]):
    """Internal helper to run anomaly detection without a separate HTTP call."""
    df = pd.DataFrame([inst.model_dump() for inst in history])
    df['rolling_mean'] = df['request_count'].rolling(window=6).mean()
    df['rolling_std'] = df['request_count'].rolling(window=6).std()
    df['delta'] = df['request_count'].diff()
    anomaly_features = ['request_count', 'rolling_mean', 'rolling_std', 'delta', 'error_5xx']
    latest_row = df.tail(1).copy()
    if latest_row[anomaly_features].isnull().any().any():
        return False, 0.0 # Not enough data to judge
    signal = anomaly_model.predict(latest_row[anomaly_features].values)
    return bool(signal[0] == -1), float(signal[0])

# --- EXISTING ENDPOINTS ---
@app.post("/predict-scaling_on_xgb", response_model=ScalingResponse)
async def predict_xgb(request: ScalingRequest):
    try:
        inference_df = engineer_features(request.history)
        X_scaled = scaler_x.transform(inference_df)
        prediction = xgb_model.predict(X_scaled)[0]
        final_forecast = max(0, prediction + NORMAL_ADJUST_COUNT)
        return {"model": "XGBoost", "forecast": round(float(final_forecast), 2), "instances": int(np.ceil(final_forecast / COUNT_PER_INSTANCE))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-scaling_on_gbr", response_model=ScalingResponse)
async def predict_gbr(request: ScalingRequest):
    try:
        inference_df = engineer_features(request.history)
        X_scaled = scaler_x.transform(inference_df)
        prediction = gbr_model.predict(X_scaled)[0]
        final_forecast = max(0, prediction + NORMAL_ADJUST_COUNT)
        return {"model": "GBR", "forecast": round(float(final_forecast), 2), "instances": int(np.ceil(final_forecast / COUNT_PER_INSTANCE))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- NEW COMBINATION ENDPOINTS ---

@app.post("/predict-scaling-smart-xgb", response_model=SmartScalingResponse)
async def predict_smart_xgb(request: ScalingRequest):
    """Combines Anomaly Detection with XGBoost Scaling."""
    try:
        is_anomaly, score = check_anomaly_internal(request.history)
        inference_df = engineer_features(request.history)
        X_scaled = scaler_x.transform(inference_df)
        prediction = xgb_model.predict(X_scaled)[0]
        
        # Logic: If anomaly is detected, we might scale more conservatively or flag a warning
        final_forecast = prediction + (ABNORMAL_ADJUST_COUNT if is_anomaly else NORMAL_ADJUST_COUNT)
        
        return {
            "recommendation": "Check system health" if is_anomaly else "Normal scaling",
            "is_anomaly": is_anomaly,
            "forecast_next_hour": round(float(final_forecast), 2),
            "recommended_instances": int(np.ceil(final_forecast / COUNT_PER_INSTANCE)),
            "model_used": "XGBoost + IsolationForest"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-scaling-smart-gbr", response_model=SmartScalingResponse)
async def predict_smart_gbr(request: ScalingRequest):
    """Combines Anomaly Detection with GBR Scaling."""
    try:
        is_anomaly, score = check_anomaly_internal(request.history)
        inference_df = engineer_features(request.history)
        X_scaled = scaler_x.transform(inference_df)
        prediction = gbr_model.predict(X_scaled)[0]
        
        final_forecast = prediction + (ABNORMAL_ADJUST_COUNT if is_anomaly else NORMAL_ADJUST_COUNT)
        
        return {
            "recommendation": "Check system health" if is_anomaly else "Normal scaling",
            "is_anomaly": is_anomaly,
            "forecast_next_hour": round(float(final_forecast), 2),
            "recommended_instances": int(np.ceil(final_forecast / COUNT_PER_INSTANCE)),
            "model_used": "GBR + IsolationForest"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: ScalingRequest):
    try:
        df = pd.DataFrame([inst.model_dump() for inst in request.history])
        
        # Isolation Forest features typically used in your pipeline
        df['rolling_mean'] = df['request_count'].rolling(window=6).mean()
        df['rolling_std'] = df['request_count'].rolling(window=6).std()
        df['delta'] = df['request_count'].diff()
        
        anomaly_features = ['request_count', 'rolling_mean', 'rolling_std', 'delta', 'error_5xx']
        latest_row = df.tail(1).copy()
        
        if latest_row[anomaly_features].isnull().any().any():
            return {"status": "pending", "message": "Need at least 6 hours for anomaly window"}

        signal = anomaly_model.predict(latest_row[anomaly_features].values)
        
        return {
            "is_anomaly": bool(signal[0] == -1),
            "anomaly_score": float(signal[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Anomaly Detection Error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "healthy"}

# --- 4. RUNNING THE SERVICE ---
if __name__ == "__main__":
    import uvicorn
    # In cloud, you'd run: uvicorn app:app --host 0.0.0.0 --port 8080
    uvicorn.run(app, host="127.0.0.1", port=8000)
