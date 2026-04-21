FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY ./model/best_gbr.pkl ./model/best_gbr.pkl
COPY ./model/best_xgb.pkl ./model/best_xgb.pkl
COPY ./model/anomaly_model.pkl ./model/anomaly_model.pkl

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
