# Capstone Project: Proactive Cloud Operations Engine
## ML-Driven Auto-Scaling

### 1. Project Vision
An end-to-end MLOps solution that transforms reactive cloud infrastructure into a proactive, self-scaling system. By analyzing historical traffic logs, the system predicts upcoming traffic and gives signal/trigger to scaling system.

---

### 2. Core Functional Pillars
* **Proactive Auto-Scaling:** Uses Time-Series Forecasting (GradientBoostingRegressor,XGBoost + IsolationForest) to anticipate traffic and adjust cloud capacity ( instance ).

---

### 3. Technology Stack
| Domain | Technologies |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-learn(GradientBoostingRegressor,IsolationForest), XGBoost |
| **Cloud Infrastructure** | GCP (Cloud Run, Artifact Registry) |
| **DevOps & MLOps** | Docker, GitHub Actions (CI/CD) |
| **Application Layer** | FastAPI (REST API), Uvicorn |

---

### 4. Implementation Phases (Milestones)

#### Phase 1: Data Engineering & EDA (2d)
* **Ingestion:** Parse [raw Apache/NASA logs](https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html)  using Regex into structured Pandas DataFrames.
* **Validation:** Clean data, handle missing values, and ensure time-series continuity.
* **EDA:** Visualize seasonal traffic patterns and error rate distributions.

#### Phase 2: Feature Engineering & Modeling(2d)
* **Temporal Features:** Implement Lag features, rolling statistics, and cyclical encoding (Sin/Cos).
* **Training:** Develop and compare RandomForest, GradientBoostingRegressor(XGBoost), and LSTM models.
* **Evaluation:** Measure performance using MAE and RMSE metrics to minimize over-provisioning.

#### Phase 3: System Architecture(3d) 
* **API Development:** Wrap models into a FastAPI service for inference.
* **Containerization:** Build optimized Docker images including the trained model artifacts.
* **Infrastructure:** Provision cloud resources (GCP) using Github Action.

#### Phase 4: Automation (CI/CD)(2d)
* **Pipeline:** Orchestrate a GitHub Actions workflow to automate training, building, and deployment.

#### Phase 5: Monitoring(2d)
* **Monitoring:** Implement health checks and logging for the deployed API.

---

## 5. API Endpoints

The FastAPI service provides 6 endpoints for traffic forecasting and anomaly detection:

### 5.1 Health Check
**GET `/health`**
- **Purpose:** Health check endpoint for service availability monitoring
- **Parameters:** None
- **Response:** `HealthResponse`
- **Example:**
  ```bash
  curl http://localhost:8000/health
  ```
  Response:
  ```json
  { "status": "healthy" }
  ```

### 5.2 XGBoost Scaling Prediction
**POST `/predict-scaling_on_xgb`**
- **Purpose:** Predicts next-hour traffic and recommended instance count using XGBoost model
- **Parameters:** `ScalingRequest` (body)
- **Response:** `ScalingResponse`
- **Model Details:** XGBoost + 10k base adjustment for normal conditions
- **Minimum History:** 25 hours of log data required
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/predict-scaling_on_xgb \
    -H "Content-Type: application/json" \
    -d '{"history": [...]}'
  ```

### 5.3 GBR (Gradient Boosting Regressor) Scaling Prediction
**POST `/predict-scaling_on_gbr`**
- **Purpose:** Predicts next-hour traffic and recommended instance count using GBR model
- **Parameters:** `ScalingRequest` (body)
- **Response:** `ScalingResponse`
- **Model Details:** GradientBoostingRegressor + 10k base adjustment for normal conditions
- **Minimum History:** 25 hours of log data required
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/predict-scaling_on_gbr \
    -H "Content-Type: application/json" \
    -d '{"history": [...]}'
  ```

### 5.4 Smart XGBoost Scaling (with Anomaly Detection)
**POST `/predict-scaling-smart-xgb`**
- **Purpose:** Combines XGBoost forecasting with IsolationForest anomaly detection for intelligent scaling recommendations
- **Parameters:** `ScalingRequest` (body)
- **Response:** `SmartScalingResponse`
- **Anomaly Adjustment:** Uses 25k adjustment if anomaly detected, 10k for normal conditions
- **Minimum History:** 25 hours required (6 hours minimum for anomaly detection window)
- **Features:**
  - Real-time anomaly detection
  - Adaptive scaling based on system health
  - Recommendation flag ("Normal scaling" or "Check system health")
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/predict-scaling-smart-xgb \
    -H "Content-Type: application/json" \
    -d '{"history": [...]}'
  ```

### 5.5 Smart GBR Scaling (with Anomaly Detection)
**POST `/predict-scaling-smart-gbr`**
- **Purpose:** Combines GBR forecasting with IsolationForest anomaly detection for intelligent scaling recommendations
- **Parameters:** `ScalingRequest` (body)
- **Response:** `SmartScalingResponse`
- **Anomaly Adjustment:** Uses 25k adjustment if anomaly detected, 10k for normal conditions
- **Minimum History:** 25 hours required (6 hours minimum for anomaly detection window)
- **Features:**
  - Real-time anomaly detection
  - Adaptive scaling based on system health
  - Recommendation flag ("Normal scaling" or "Check system health")

### 5.6 Smart XGBoost + GBR Scaling (with Anomaly Detection)
**POST `/predict-scaling-smart`**
- **Purpose:** Combines Average(XGBoost + GBR forecasting) with IsolationForest anomaly detection for intelligent scaling recommendations
- **Parameters:** `ScalingRequest` (body)
- **Response:** `SmartScalingResponse`
- **Anomaly Adjustment:** Uses 25k adjustment if anomaly detected, 10k for normal conditions
- **Minimum History:** 25 hours required (6 hours minimum for anomaly detection window)
- **Features:**
  - Real-time anomaly detection
  - Adaptive scaling based on system health
  - Recommendation flag ("Normal scaling" or "Check system health")
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/predict-scaling-smart-xgb \
    -H "Content-Type: application/json" \
    -d '{"history": [...]}'
  ```

### 5.7 Anomaly Detection Only
**POST `/detect-anomalies`**
- **Purpose:** Standalone anomaly detection using IsolationForest without scaling prediction
- **Parameters:** `ScalingRequest` (body)
- **Response:** `AnomalyDetectionResponse`
- **Use Case:** Identify unusual traffic patterns independently
- **Status Values:**
  - `success`: Anomaly detection completed
  - `pending`: Insufficient data for rolling window (needs 6+ hours)
- **Example:**
  ```bash
  curl -X POST http://localhost:8000/detect-anomalies \
    -H "Content-Type: application/json" \
    -d '{"history": [...]}'
  ```

---

## 6. Data Schemas

### Request Schemas

#### ScalingRequest
Used by all scaling and anomaly detection endpoints.

```python
{
  "history": [LogInstance, ...]  # List of hourly log data (minimum 25 hours)
}
```

#### LogInstance
Represents one hour of traffic metrics.

| Field | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| `request_count` | int | ≥ 0 | Number of HTTP requests in the hour |
| `error_5xx` | int | ≥ 0 | Count of 5xx server errors |
| `bytes_sum` | int | ≥ 0 | Total bytes transferred (sum of response sizes) |
| `hour` | int | 0-23 | Hour of the day in 24-hour format |

**Example:**
```json
{
  "history": [
    {
      "request_count": 1050,
      "error_5xx": 12,
      "bytes_sum": 5200,
      "hour": 0
    },
    {
      "request_count": 1100,
      "error_5xx": 15,
      "bytes_sum": 5400,
      "hour": 1
    }
  ]
}
```

### Response Schemas

#### ScalingResponse
Response from basic scaling endpoints (`/predict-scaling_on_xgb`, `/predict-scaling_on_gbr`).

| Field | Type | Description |
| :--- | :--- | :--- |
| `model` | str | Model used ("XGBoost" or "GBR") |
| `forecast` | float | Predicted request count for next hour |
| `instances` | int | Recommended number of instances (based on `COUNT_PER_INSTANCE=100000`) |

**Example:**
```json
{
  "model": "XGBoost",
  "forecast": 11500.25,
  "instances": 1
}
```

#### SmartScalingResponse
Response from smart endpoints (`/predict-scaling-smart`,`/predict-scaling-smart-xgb`, `/predict-scaling-smart-gbr`).

| Field | Type | Description |
| :--- | :--- | :--- |
| `recommendation` | str | "Normal scaling" or "Check system health" (based on anomaly detection) |
| `is_anomaly` | bool | True if unusual traffic pattern detected |
| `forecast_next_hour` | float | Predicted request count for next hour |
| `recommended_instances` | int | Recommended instance count (adjusted for anomalies) |
| `model_used` | str | Model identifier ("XGBoost + IsolationForest" or "GBR + IsolationForest") |

**Example:**
```json
{
  "recommendation": "Normal scaling",
  "is_anomaly": false,
  "forecast_next_hour": 11500.25,
  "recommended_instances": 1,
  "model_used": "XGBoost + IsolationForest"
}
```

#### AnomalyDetectionResponse
Response from `/detect-anomalies` endpoint.

| Field | Type | Description |
| :--- | :--- | :--- |
| `is_anomaly` | bool \| null | True if anomaly detected, null if pending |
| `anomaly_score` | float \| null | Anomaly score from IsolationForest, null if pending |
| `status` | str | "success" or "pending" |
| `message` | str \| null | Status message (e.g., "Need at least 6 hours for anomaly window") |

**Example (Success):**
```json
{
  "is_anomaly": false,
  "anomaly_score": 1,
  "status": "success",
  "message": null
}
```

**Example (Pending - Insufficient Data):**
```json
{
  "is_anomaly": null,
  "anomaly_score": null,
  "status": "pending",
  "message": "Need at least 6 hours for anomaly window"
}
```

#### HealthResponse
Response from `/health` endpoint.

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | str | Service status ("healthy") |

---

## 7. Feature Engineering

All prediction endpoints use the following engineered features from historical log data:

| Feature | Description |
| :--- | :--- |
| `request_count` | Current hourly request count |
| `error_5xx` | Current hourly 5xx error count |
| `bytes_sum` | Current hourly bytes transferred |
| `lag_24h` | Request count from 24 hours ago |
| `lag_1h` | Request count from 1 hour ago |
| `lag_2h` | Request count from 2 hours ago |
| `rolling_mean_3h` | 3-hour rolling average of requests |
| `velocity` | Rate of change in request count |
| `hour_sin` | Cyclical encoding of hour (sine) |
| `hour_cos` | Cyclical encoding of hour (cosine) |

**Anomaly Detection Features:**
- `rolling_mean` (6-hour window)
- `rolling_std` (6-hour window)
- `delta` (hourly change)
- `request_count`
- `error_5xx`

---

## 8. Error Handling

All endpoints return structured error responses:

**Validation Error (400):**
```json
{
  "detail": "Validation error",
  "errors": [
    {
      "field": "['request_count']",
      "message": "Input should be greater than or equal to 0",
      "type": "greater_than_equal"
    }
  ]
}
```

**Insufficient History (400):**
```json
{
  "detail": "Insufficient history. Provide 25+ hours of data."
}
```

**Model Prediction Error (400):**
```json
{
  "detail": "Error message from model inference"
}
```

---

## 9. Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `COUNT_PER_INSTANCE` | 100,000 | Requests per instance (used to calculate `instances` field) |
| `NORMAL_ADJUST_COUNT` | 10,000 | Base forecast adjustment during normal conditions |
| `ABNORMAL_ADJUST_COUNT` | 25,000 | Enhanced forecast adjustment during anomalies |

