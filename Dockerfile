FROM python:3.13-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY main.py .
COPY src/ src/
COPY model/ model/

# Install production dependencies
RUN pip install --no-cache-dir -e ".[prod]"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
