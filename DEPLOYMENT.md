# Model Creation
- Install dependencies
  ```
  pip install --no-cache-dir -e ".[prod]"
  ```
- Train and create model
  ```
  python3 train_and_save.py
  ```
# Docker
- Image Creation
  ```
  docker build -t instances_prediction_model:v1 .
  ```
- Docker Instance
  ```
  docker run -p 8080:8080 instances_prediction_model:v1
  ```
# Application
- Test Request(Local,Docker)
  - Anomaly Endpoint
    ```
    curl -X 'POST' \
    'http://localhost:8080/detect-anomalies' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
  - Scaling Endpoint
    ```
    curl -X 'POST' \
    'http://localhost:8080/predict-scaling_on_xgb' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://localhost:8080/predict-scaling_on_gbr' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
  - Scaling + Anomaly Endpoint
    ```
    curl -X 'POST' \
    'http://localhost:8080/predict-scaling-smart' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://localhost:8080/predict-scaling-smart-gbr' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://localhost:8080/predict-scaling-smart-xgb' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
- CloudRun Test HostName ex. scalingpredict-402889997289.us-west1.run.app
  - Anomaly Endpoint
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/detect-anomalies' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
    {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
    {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
    {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
    {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
    {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
    {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
    {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
    {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
    {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
    {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
    {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
    {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
    {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
    {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
    {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
    {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
    {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
    {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
    {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
    {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
    {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
    {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
    {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
    {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
    {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
  - Scaling Endpoint
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/predict-scaling_on_xgb' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/predict-scaling_on_gbr' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
  - Scaling + Anomaly Endpoint
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/predict-scaling-smart' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/predict-scaling-smart-gbr' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
    ```
    curl -X 'POST' \
    'http://scalingpredict-402889997289.us-west1.run.app/predict-scaling-smart-xgb' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "history": [
      {"request_count": 450, "error_5xx": 2, "bytes_sum": 150000, "hour": 0},
      {"request_count": 380, "error_5xx": 0, "bytes_sum": 120000, "hour": 1},
      {"request_count": 310, "error_5xx": 1, "bytes_sum": 95000, "hour": 2},
      {"request_count": 280, "error_5xx": 0, "bytes_sum": 80000, "hour": 3},
      {"request_count": 220, "error_5xx": 0, "bytes_sum": 65000, "hour": 4},
      {"request_count": 190, "error_5xx": 4, "bytes_sum": 50000, "hour": 5},
      {"request_count": 180, "error_5xx": 1, "bytes_sum": 45000, "hour": 6},
      {"request_count": 210, "error_5xx": 0, "bytes_sum": 55000, "hour": 7},
      {"request_count": 250, "error_5xx": 0, "bytes_sum": 70000, "hour": 8},
      {"request_count": 320, "error_5xx": 0, "bytes_sum": 90000, "hour": 9},
      {"request_count": 410, "error_5xx": 2, "bytes_sum": 130000, "hour": 10},
      {"request_count": 550, "error_5xx": 3, "bytes_sum": 180000, "hour": 11},
      {"request_count": 680, "error_5xx": 1, "bytes_sum": 250000, "hour": 12},
      {"request_count": 820, "error_5xx": 0, "bytes_sum": 310000, "hour": 13},
      {"request_count": 950, "error_5xx": 5, "bytes_sum": 400000, "hour": 14},
      {"request_count": 1100, "error_5xx": 2, "bytes_sum": 450000, "hour": 15},
      {"request_count": 1250, "error_5xx": 0, "bytes_sum": 520000, "hour": 16},
      {"request_count": 1400, "error_5xx": 8, "bytes_sum": 600000, "hour": 17},
      {"request_count": 1380, "error_5xx": 4, "bytes_sum": 580000, "hour": 18},
      {"request_count": 1200, "error_5xx": 1, "bytes_sum": 500000, "hour": 19},
      {"request_count": 950, "error_5xx": 0, "bytes_sum": 410000, "hour": 20},
      {"request_count": 720, "error_5xx": 2, "bytes_sum": 300000, "hour": 21},
      {"request_count": 600, "error_5xx": 0, "bytes_sum": 240000, "hour": 22},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 23},
      {"request_count": 520, "error_5xx": 1, "bytes_sum": 190000, "hour": 0}
      ]
    }'
    ```
# Production Deployment(Github action main puxsh)
- Model upload
  ```
  gcloud storage cp model/* gs://{{bucket_name}}/model/
  ```
- Deployment(Github action)