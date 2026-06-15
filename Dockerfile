ARG ZKLLM_IMAGE_PLATFORM=linux/amd64
FROM --platform=${ZKLLM_IMAGE_PLATFORM} python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HOME=/root/.cache/huggingface \
    ZKLLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    ZKLLM_LOG_LEVEL=INFO

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY server/requirements.txt /app/server-requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0,<2.9" \
    && pip install -r /app/requirements.txt -r /app/server-requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1

CMD ["python", "-m", "uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
