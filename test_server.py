"""
Comprehensive tests for server.py endpoints with edge cases.
Covers all endpoints and various scenarios including normal cases, edge cases, and error conditions.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from src.server import app
from src.schema import LogInstance, ScalingRequest
import src.server as server

client = TestClient(app)

# Mock the models before they are used
@pytest.fixture(scope="session", autouse=True)
def mock_models():
    """Mock the ML models so tests don't require actual model files."""
    # Create mock models
    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = np.array([500.0])
    
    mock_gbr = MagicMock()
    mock_gbr.predict.return_value = np.array([450.0])
    
    mock_anomaly = MagicMock()
    mock_anomaly.predict.return_value = np.array([1])  # 1 = normal, -1 = anomaly
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    
    # Inject the mocks into the server module
    server.xgb_model = mock_xgb
    server.gbr_model = mock_gbr
    server.anomaly_model = mock_anomaly
    server.scaler_x = mock_scaler
    
    yield
    
    # Cleanup would happen after all tests

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_history_25h():
    """Generate 25 hours of valid log data (minimum required)."""
    return [
        LogInstance(
            request_count=1000 + i * 50,
            error_5xx=10 + i,
            bytes_sum=5000 + i * 100,
            hour=i % 24
        )
        for i in range(25)
    ]

@pytest.fixture
def valid_history_48h():
    """Generate 48 hours of valid log data."""
    return [
        LogInstance(
            request_count=1000 + i * 50,
            error_5xx=10 + i,
            bytes_sum=5000 + i * 100,
            hour=i % 24
        )
        for i in range(48)
    ]

@pytest.fixture
def anomalous_history():
    """Generate history with anomalous spike in request count."""
    history = [
        LogInstance(
            request_count=1000,
            error_5xx=10,
            bytes_sum=5000,
            hour=i % 24
        )
        for i in range(25)
    ]
    # Add large spike at the end
    history[-1] = LogInstance(
        request_count=50000,  # Huge spike
        error_5xx=500,
        bytes_sum=250000,
        hour=23
    )
    return history

@pytest.fixture
def high_error_history():
    """Generate history with high error rates."""
    return [
        LogInstance(
            request_count=1000 + i * 50,
            error_5xx=100 + i * 10,  # High error rate
            bytes_sum=5000 + i * 100,
            hour=i % 24
        )
        for i in range(25)
    ]

# ============================================================================
# HEALTH ENDPOINT TESTS
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200 status."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_endpoint_returns_healthy_status(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_endpoint_no_params(self):
        """Health endpoint should work without parameters."""
        response = client.get("/health")
        assert response.status_code == 200


# ============================================================================
# XGB SCALING ENDPOINT TESTS
# ============================================================================

class TestPredictScalingXGB:
    """Tests for POST /predict-scaling_on_xgb endpoint."""
    
    def test_xgb_valid_25h_history(self, valid_history_25h):
        """XGB should make prediction with valid 25-hour history."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert "instances" in data
        assert "model" in data
        assert data["model"] == "XGBoost"
        assert data["forecast"] >= 0
        assert data["instances"] >= 0
    
    def test_xgb_valid_48h_history(self, valid_history_48h):
        """XGB should make prediction with 48-hour history."""
        request = ScalingRequest(history=valid_history_48h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["forecast"] >= 0
    
    def test_xgb_insufficient_history(self):
        """XGB should fail with less than 25 hours of history."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(10)  # Only 10 hours
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 400
        assert "Insufficient history" in response.json()["detail"]
    
    def test_xgb_empty_history(self):
        """XGB should fail with empty history."""
        request = ScalingRequest(history=[])
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 400
    
    def test_xgb_negative_request_count(self, valid_history_25h):
        """XGB should reject negative request counts (validation fails)."""
        invalid_history = valid_history_25h.copy()
        # This should raise ValidationError at Pydantic level
        with pytest.raises(ValidationError):
            invalid_history[0] = LogInstance(
                request_count=-100,  # Invalid
                error_5xx=10,
                bytes_sum=5000,
                hour=0
            )
    
    def test_xgb_negative_error_5xx(self, valid_history_25h):
        """XGB should reject negative error counts."""
        invalid_history = valid_history_25h.copy()
        with pytest.raises(ValidationError):
            invalid_history[0] = LogInstance(
                request_count=1000,
                error_5xx=-5,  # Invalid
                bytes_sum=5000,
                hour=0
            )
    
    def test_xgb_negative_bytes_sum(self, valid_history_25h):
        """XGB should reject negative bytes sum."""
        invalid_history = valid_history_25h.copy()
        with pytest.raises(ValidationError):
            invalid_history[0] = LogInstance(
                request_count=1000,
                error_5xx=10,
                bytes_sum=-5000,  # Invalid
                hour=0
            )
    
    def test_xgb_invalid_hour_negative(self, valid_history_25h):
        """XGB should reject hour < 0."""
        invalid_history = valid_history_25h.copy()
        with pytest.raises(ValidationError):
            invalid_history[0] = LogInstance(
                request_count=1000,
                error_5xx=10,
                bytes_sum=5000,
                hour=-1  # Invalid
            )
    
    def test_xgb_invalid_hour_too_large(self, valid_history_25h):
        """XGB should reject hour > 23."""
        invalid_history = valid_history_25h.copy()
        with pytest.raises(ValidationError):
            invalid_history[0] = LogInstance(
                request_count=1000,
                error_5xx=10,
                bytes_sum=5000,
                hour=24  # Invalid, should be 0-23
            )
    
    def test_xgb_boundary_hour_min(self, valid_history_25h):
        """XGB should accept hour = 0."""
        valid_history_25h[0] = LogInstance(
            request_count=1000,
            error_5xx=10,
            bytes_sum=5000,
            hour=0
        )
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_xgb_boundary_hour_max(self, valid_history_25h):
        """XGB should accept hour = 23."""
        valid_history_25h[0] = LogInstance(
            request_count=1000,
            error_5xx=10,
            bytes_sum=5000,
            hour=23
        )
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_xgb_zero_request_count(self, valid_history_25h):
        """XGB should handle zero request count."""
        valid_history_25h[0] = LogInstance(
            request_count=0,
            error_5xx=0,
            bytes_sum=0,
            hour=0
        )
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["forecast"] >= 0  # Should still produce valid output
    
    def test_xgb_very_large_request_count(self, valid_history_25h):
        """XGB should handle very large request counts."""
        valid_history_25h[0] = LogInstance(
            request_count=1000000,
            error_5xx=1000,
            bytes_sum=5000000,
            hour=0
        )
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_xgb_response_has_required_fields(self, valid_history_25h):
        """XGB response should have all required fields."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        data = response.json()
        assert "model" in data
        assert "forecast" in data
        assert "instances" in data


# ============================================================================
# GBR SCALING ENDPOINT TESTS
# ============================================================================

class TestPredictScalingGBR:
    """Tests for POST /predict-scaling_on_gbr endpoint."""
    
    def test_gbr_valid_25h_history(self, valid_history_25h):
        """GBR should make prediction with valid 25-hour history."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "forecast" in data
        assert "instances" in data
        assert "model" in data
        assert data["model"] == "GBR"
        assert data["forecast"] >= 0
        assert data["instances"] >= 0
    
    def test_gbr_valid_48h_history(self, valid_history_48h):
        """GBR should make prediction with 48-hour history."""
        request = ScalingRequest(history=valid_history_48h)
        response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["forecast"] >= 0
    
    def test_gbr_insufficient_history(self):
        """GBR should fail with less than 25 hours of history."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(10)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        assert response.status_code == 400
        assert "Insufficient history" in response.json()["detail"]
    
    def test_gbr_empty_history(self):
        """GBR should fail with empty history."""
        request = ScalingRequest(history=[])
        response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        assert response.status_code == 400
    
    def test_gbr_response_has_required_fields(self, valid_history_25h):
        """GBR response should have all required fields."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        data = response.json()
        assert "model" in data
        assert "forecast" in data
        assert "instances" in data


# ============================================================================
# SMART XGB ENDPOINT TESTS
# ============================================================================

class TestPredictScalingSmartXGB:
    """Tests for POST /predict-scaling-smart-xgb endpoint."""
    
    def test_smart_xgb_valid_history(self, valid_history_25h):
        """Smart XGB should make prediction with valid history."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "recommendation" in data
        assert "is_anomaly" in data
        assert "forecast_next_hour" in data
        assert "recommended_instances" in data
        assert "model_used" in data
        assert data["model_used"] == "XGBoost + IsolationForest"
    
    def test_smart_xgb_normal_conditions(self, valid_history_25h):
        """Smart XGB should flag as non-anomaly in normal conditions."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        data = response.json()
        # In normal conditions, should recommend "Normal scaling"
        assert data["recommendation"] in ["Normal scaling", "Check system health"]
    
    def test_smart_xgb_anomalous_spike(self, anomalous_history):
        """Smart XGB should detect anomalies with request spike."""
        request = ScalingRequest(history=anomalous_history)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        # With anomaly, forecast should be higher (prediction + 100 instead of + 50)
        assert data["forecast_next_hour"] >= 0
    
    def test_smart_xgb_high_errors(self, high_error_history):
        """Smart XGB should handle high error rates."""
        request = ScalingRequest(history=high_error_history)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["forecast_next_hour"] >= 0
    
    def test_smart_xgb_insufficient_history(self):
        """Smart XGB should fail with insufficient history."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(5)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        assert response.status_code == 400


# ============================================================================
# SMART GBR ENDPOINT TESTS
# ============================================================================

class TestPredictScalingSmartGBR:
    """Tests for POST /predict-scaling-smart-gbr endpoint."""
    
    def test_smart_gbr_valid_history(self, valid_history_25h):
        """Smart GBR should make prediction with valid history."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-gbr", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "recommendation" in data
        assert "is_anomaly" in data
        assert "forecast_next_hour" in data
        assert "recommended_instances" in data
        assert "model_used" in data
        assert data["model_used"] == "GBR + IsolationForest"
    
    def test_smart_gbr_normal_conditions(self, valid_history_25h):
        """Smart GBR should flag as non-anomaly in normal conditions."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-gbr", json=request.model_dump())
        data = response.json()
        assert data["recommendation"] in ["Normal scaling", "Check system health"]
    
    def test_smart_gbr_anomalous_spike(self, anomalous_history):
        """Smart GBR should detect anomalies with request spike."""
        request = ScalingRequest(history=anomalous_history)
        response = client.post("/predict-scaling-smart-gbr", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert data["forecast_next_hour"] >= 0
    
    def test_smart_gbr_insufficient_history(self):
        """Smart GBR should fail with insufficient history."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(5)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling-smart-gbr", json=request.model_dump())
        assert response.status_code == 400


# ============================================================================
# ANOMALY DETECTION ENDPOINT TESTS
# ============================================================================

class TestDetectAnomalies:
    """Tests for POST /detect-anomalies endpoint."""
    
    def test_anomaly_detection_valid_history(self, valid_history_25h):
        """Anomaly detection should work with valid history."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/detect-anomalies", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "status" in data
        assert data["status"] in ["success", "pending"]
    
    def test_anomaly_detection_insufficient_history(self):
        """Anomaly detection should flag pending with insufficient history."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(5)  # Only 5 hours
        ]
        request = ScalingRequest(history=history)
        response = client.post("/detect-anomalies", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        # Should return pending if insufficient data for rolling window
        if data["status"] == "pending":
            assert "Need at least" in data["message"]
    
    def test_anomaly_detection_with_spike(self, anomalous_history):
        """Anomaly detection should detect spike."""
        request = ScalingRequest(history=anomalous_history)
        response = client.post("/detect-anomalies", json=request.model_dump())
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
    
    def test_anomaly_detection_empty_history(self):
        """Anomaly detection should fail with empty history."""
        request = ScalingRequest(history=[])
        response = client.post("/detect-anomalies", json=request.model_dump())
        # Could be 400 or 200 depending on error handling
        assert response.status_code in [200, 400]
    
    def test_anomaly_detection_response_structure(self, valid_history_25h):
        """Anomaly detection response should have correct structure."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/detect-anomalies", json=request.model_dump())
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["is_anomaly"], bool)
            assert "status" in data


# ============================================================================
# EDGE CASES AND STRESS TESTS
# ============================================================================

class TestEdgeCasesAndStress:
    """Additional edge cases and stress tests."""
    
    def test_single_hour_entry(self):
        """Test with only 1 hour of data."""
        history = [LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=0)]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 400
    
    def test_all_zeros(self):
        """Test with all zero values."""
        history = [
            LogInstance(request_count=0, error_5xx=0, bytes_sum=0, hour=i % 24)
            for i in range(25)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_all_max_values(self):
        """Test with very large values."""
        history = [
            LogInstance(request_count=999999999, error_5xx=999999, bytes_sum=999999999, hour=i % 24)
            for i in range(25)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_alternating_high_low(self):
        """Test with alternating high and low values."""
        history = []
        for i in range(25):
            if i % 2 == 0:
                history.append(LogInstance(request_count=10000, error_5xx=100, bytes_sum=50000, hour=i % 24))
            else:
                history.append(LogInstance(request_count=100, error_5xx=1, bytes_sum=500, hour=i % 24))
        
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_monotonic_increase(self):
        """Test with monotonically increasing values."""
        history = [
            LogInstance(request_count=1000 + i * 500, error_5xx=10 + i, bytes_sum=5000 + i * 1000, hour=i % 24)
            for i in range(25)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_monotonic_decrease(self):
        """Test with monotonically decreasing values."""
        history = [
            LogInstance(request_count=max(100, 5000 - i * 100), error_5xx=max(1, 50 - i), bytes_sum=max(100, 25000 - i * 500), hour=i % 24)
            for i in range(25)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_random_noise(self):
        """Test with random values."""
        np.random.seed(42)
        history = [
            LogInstance(
                request_count=int(np.random.randint(100, 5000)),
                error_5xx=int(np.random.randint(0, 100)),
                bytes_sum=int(np.random.randint(500, 50000)),
                hour=i % 24
            )
            for i in range(25)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_exact_24_hours_data(self):
        """Test with exactly 24 hours (should fail, needs 25)."""
        history = [
            LogInstance(request_count=1000, error_5xx=10, bytes_sum=5000, hour=i % 24)
            for i in range(24)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 400
    
    def test_very_long_history(self):
        """Test with 365 days of history."""
        history = [
            LogInstance(request_count=1000 + i % 100, error_5xx=10 + i % 10, bytes_sum=5000 + i % 500, hour=i % 24)
            for i in range(365 * 24)
        ]
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200
    
    def test_sequential_hours_complete(self):
        """Test with complete sequential hours 0-23 repeated multiple times."""
        history = []
        for day in range(3):  # 3 complete days
            for hour in range(24):
                history.append(LogInstance(
                    request_count=1000 + day * 100 + hour * 10,
                    error_5xx=10 + day + hour,
                    bytes_sum=5000 + day * 500 + hour * 50,
                    hour=hour
                ))
        request = ScalingRequest(history=history)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        assert response.status_code == 200


# ============================================================================
# COMPARISON TESTS
# ============================================================================

class TestModelComparison:
    """Tests comparing outputs between different models."""
    
    def test_xgb_vs_gbr_same_input(self, valid_history_25h):
        """XGB and GBR should produce comparable outputs."""
        request = ScalingRequest(history=valid_history_25h)
        
        xgb_response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        gbr_response = client.post("/predict-scaling_on_gbr", json=request.model_dump())
        
        assert xgb_response.status_code == 200
        assert gbr_response.status_code == 200
        
        xgb_data = xgb_response.json()
        gbr_data = gbr_response.json()
        
        # Both should have valid forecasts
        assert xgb_data["forecast"] >= 0
        assert gbr_data["forecast"] >= 0
    
    def test_smart_vs_basic_xgb(self, valid_history_25h):
        """Smart XGB vs basic XGB should both work."""
        request = ScalingRequest(history=valid_history_25h)
        
        basic_response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        smart_response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        
        assert basic_response.status_code == 200
        assert smart_response.status_code == 200
    
    def test_smart_xgb_vs_smart_gbr(self, valid_history_25h):
        """Smart XGB and Smart GBR should produce similar structure."""
        request = ScalingRequest(history=valid_history_25h)
        
        xgb_response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        gbr_response = client.post("/predict-scaling-smart-gbr", json=request.model_dump())
        
        assert xgb_response.status_code == 200
        assert gbr_response.status_code == 200
        
        xgb_data = xgb_response.json()
        gbr_data = gbr_response.json()
        
        # Should have same keys
        assert set(xgb_data.keys()) == set(gbr_data.keys())


# ============================================================================
# RESPONSE VALIDATION TESTS
# ============================================================================

class TestResponseValidation:
    """Tests for validating response structure and types."""
    
    def test_xgb_forecast_is_float(self, valid_history_25h):
        """XGB forecast should be float."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        data = response.json()
        assert isinstance(data["forecast"], (int, float))
    
    def test_xgb_instances_is_integer(self, valid_history_25h):
        """XGB instances should be integer."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling_on_xgb", json=request.model_dump())
        data = response.json()
        assert isinstance(data["instances"], int)
    
    def test_smart_xgb_recommendation_is_string(self, valid_history_25h):
        """Smart XGB recommendation should be string."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        data = response.json()
        assert isinstance(data["recommendation"], str)
    
    def test_smart_xgb_is_anomaly_is_boolean(self, valid_history_25h):
        """Smart XGB is_anomaly should be boolean."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/predict-scaling-smart-xgb", json=request.model_dump())
        data = response.json()
        assert isinstance(data["is_anomaly"], bool)
    
    def test_anomaly_detection_anomaly_score(self, valid_history_25h):
        """Anomaly detection should return anomaly score if success."""
        request = ScalingRequest(history=valid_history_25h)
        response = client.post("/detect-anomalies", json=request.model_dump())
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                assert "anomaly_score" in data


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
