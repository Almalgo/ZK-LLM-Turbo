# Deployment: Coolify API and SNET Publisher

This repo supports two deployment targets from the same codebase. The current
production path is Daemon Only HaaS with the FastAPI runtime hosted on Coolify.
See `docs/deploy-coolify-haas.md`.

1. Coolify-hosted FastAPI service runtime.
2. SingularityNET Publisher Daemon Only HaaS.
3. Full-Stack HaaS lightweight proxy remains documented for rollback/history.

## Coolify FastAPI deployment

Recommended Coolify settings:

- Build pack: Dockerfile
- Dockerfile: `Dockerfile.fastapi`
- Public port: `8000`
- Healthcheck path: `/heartbeat`
- Environment:
  - `PORT=8000`
  - `ZKLLM_SERVER_MODEL_DTYPE=float32`

Do not use the root `Dockerfile` for Coolify. The root `Dockerfile` is the
Full-Stack HaaS image and starts `runpod_handler.py`.

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

## SNET Publisher / Daemon Only HaaS

Use Publisher Daemon Only for `almalgo_labs / zk_llm2`.

```text
Service endpoint: https://zkllm.almalgo.com
Heartbeat endpoint: https://zkllm.almalgo.com/heartbeat
Service heartbeat type: http
```

The daemon heartbeat docs require `heartbeat_endpoint` to be a valid
`http|https|grpc` URL. If Publisher does not expose that field, support must set
it server-side; otherwise daemon logs show `serviceHeartbeatURL: ""` and the
marketplace will mark the service offline.

Full-Stack HaaS uses the root `Dockerfile` and starts:

```text
python -u runpod_handler.py
```

For Publisher metadata, upload/use:

- Proto: `snet_service/proto/zk_llm_http_api.proto`
- Package/service: `zk_llm.ZKLLMService`
- HTTP mappings:
  - `health` -> `POST /health`
  - `session` -> `POST /session`
  - `layer` -> `POST /layer`

The Publisher-hosted HaaS service forwards session/layer requests to the
self-hosted FastAPI backend. The FastAPI backend implements matching HTTP routes
for direct preflight and optional rollback testing.

## Local verification

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest requests-mock
pytest -q tests/test_customer_main.py server/tests/test_api_endpoint.py
PORT=8000 uvicorn server.server:app --host 127.0.0.1 --port 8000
```

In another shell:

```bash
python scripts/snet_endpoint_preflight.py \
  --public-base-url http://127.0.0.1:8000 \
  --allow-http
```
