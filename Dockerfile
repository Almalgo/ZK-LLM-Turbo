FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV ZKLLM_SERVER_MODEL_DTYPE=float32

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.4.1+cpu" \
    && pip install -r /app/server/requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8000}/heartbeat || exit 1

CMD ["sh", "-c", "uvicorn server.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
