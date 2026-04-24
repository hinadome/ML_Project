"""
Pydantic schema definitions for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional

class LogInstance(BaseModel):
    """Log instance data for a single hour."""
    request_count: int = Field(..., ge=0, description="Number of requests")
    error_5xx: int = Field(..., ge=0, description="Number of 5xx errors")
    bytes_sum: int = Field(..., ge=0, description="Total bytes transferred")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")

class ScalingRequest(BaseModel):
    """Request body for scaling prediction endpoints."""
    history: List[LogInstance] = Field(..., description="Historical log data (minimum 25 hours)")

# --- Response Models ---

class ScalingResponse(BaseModel):
    """Response for /predict-scaling_on_xgb and /predict-scaling_on_gbr"""
    model: str
    forecast: float
    instances: int

class SmartScalingResponse(BaseModel):
    """Response for /predict-scaling-smart-xgb and /predict-scaling-smart-gbr"""
    recommendation: str
    is_anomaly: bool
    forecast_next_hour: float
    recommended_instances: int
    model_used: str

class AnomalyDetectionResponse(BaseModel):
    """Response for /detect-anomalies"""
    is_anomaly: Optional[bool] = None
    anomaly_score: Optional[float] = None
    status: str
    message: Optional[str] = None

class HealthResponse(BaseModel):
    """Response for /health"""
    status: str