# syntax=docker/dockerfile:1
# Multi-purpose image: serves the FastAPI API by default.
# Build:  docker build -t heart-api .
# Run:    docker run -p 8000:8000 heart-api
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, data, and the committed model artifact.
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

# Train at build time if no model was committed (idempotent safety net).
RUN python -c "from pathlib import Path; import sys; \
sys.exit(0) if Path('models/heart_pipeline.joblib').exists() else None" \
    || python -m heart.train

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
