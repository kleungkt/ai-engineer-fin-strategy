# Unified Dockerfile — build any project via build arg
# Usage: docker build --build-arg PROJECT=01-nl-stock-query -t p1-api .

ARG PROJECT=01-nl-stock-query

FROM python:3.12-slim

ARG PROJECT

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project requirements and install
COPY projects/${PROJECT}/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn

# Copy source code
COPY projects/${PROJECT}/src /app/src
COPY shared /app/shared

ENV PYTHONPATH=/app/src
ENV PROJECT=${PROJECT}

EXPOSE 8000

# Auto-detect which api.py to run
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000"]
