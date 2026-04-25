FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi[standard]" \
    pandas \
    numpy \
    joblib \
    scikit-learn \
    xgboost

COPY main.py .
COPY src/ src/
COPY model/ model/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
