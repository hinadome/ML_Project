FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi[standard]" \
    pandas \
    numpy \
    joblib \
    scikit-learn \
    xgboost

COPY server.py .
COPY ./model/best_gbr.pkl ./model/best_gbr.pkl
COPY ./model/best_xgb.pkl ./model/best_xgb.pkl
COPY ./model/anomaly_model.pkl ./model/anomaly_model.pkl
COPY ./model/scaler_x.pkl ./model/scaler_x.pkl

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
