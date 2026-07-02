# Deployment: Coolify API and SNET Publisher

This repo supports two deployment targets from the same codebase:

1. Coolify-hosted FastAPI service for `https://zkllm.almalgo.com/`.
2. SingularityNET Publisher / Full-Stack HaaS with SNET-hosted daemon.

## Coolify FastAPI deployment

Recommended Coolify settings:

- Build pack: Dockerfile
- Dockerfile: `Dockerfile.fastapi`
- Public port: `8000`
- Healthcheck path: `/heartbeat`
- Environment:
  - `PORT=8000`
  - `ZKLLM_SERVER_MODEL_DTYPE=float32`

Alternative root-image mode:

- Dockerfile: `Dockerfile`
- Environment:
  - `SERVICE_MODE=fastapi`
  - `PORT=8000`
  - `ZKLLM_SERVER_MODEL_DTYPE=float32`

Expected public checks after deployment:

```bash
curl https://zkllm.almalgo.com/
curl https://zkllm.almalgo.com/heartbeat
curl https://zkllm.almalgo.com/health
curl -X POST https://zkllm.almalgo.com/health \
  -H 'Content-Type: application/json' \
  -d '{"op":"health"}'
```

You can also run the bundled preflight:

```bash
python scripts/snet_endpoint_preflight.py \
  --public-base-url https://zkllm.almalgo.com
```

If these commands return `503 no available server`, Coolify routing has no healthy backend container. Check the Coolify deployment logs, selected Dockerfile, public port, and healthcheck path.

If TLS verification fails, replace the self-signed certificate with a valid public certificate before using the URL in SNET Publisher.

## SNET Publisher / Full-Stack HaaS

Use the root `Dockerfile` for SNET Full-Stack HaaS. With no `PORT` and no `SERVICE_MODE`, it defaults to RunPod/HaaS mode:

```text
python -u runpod_handler.py
```

The HaaS wrapper calls `customer_main.run(input_data)`. `profile.json` is intentionally lightweight and exercises the health path:

```json
{"input":{"op":"health"}}
```

For Publisher metadata, upload/use:

- Proto: `snet_service/proto/zk_llm_http_api.proto`
- Package/service: `zk_llm.ZKLLMService`
- HTTP mappings:
  - `Health` -> `POST /health`
  - `Session` -> `POST /api/session`
  - `Layer` -> `POST /api/layer`

The FastAPI service implements matching HTTP routes so Publisher/daemon passthrough checks can probe health before encrypted inference fixtures are available.

## Local verification

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest requests-mock
pytest -q tests/test_customer_main.py server/tests/test_api_endpoint.py
SERVICE_MODE=fastapi PORT=8000 uvicorn server.server:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
python scripts/snet_endpoint_preflight.py \
  --public-base-url http://127.0.0.1:8000 \
  --allow-http
```
