FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUTF8=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# models/ dir created at runtime by trainer; api loads from MLflow registry
RUN mkdir -p /app/models
