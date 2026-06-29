FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV ZKLLM_SERVER_MODEL_DTYPE=float32
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x /app/docker-entrypoint.sh \
    && python -c "import fastapi, uvicorn, runpod, tenseal, torch, transformers; print('runtime imports ok')"

EXPOSE 8000

CMD ["/app/docker-entrypoint.sh"]
