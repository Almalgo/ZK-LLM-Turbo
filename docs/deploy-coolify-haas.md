# Legacy: Coolify Service + SingularityNET HAAS Daemon Deployment

This split deployment remains documented for rollback or debugging, but it is
not the preferred production path. Use `docs/deploy-full-haas.md` when both the
daemon and service runtime should be hosted by SingularityNET Publisher.

Legacy split deployment shape:

- Coolify runs the FastAPI service runtime.
- SingularityNET Publisher HAAS runs only the daemon.
- The HAAS daemon forwards HTTP passthrough requests to the Coolify public base URL.

## Mainnet identifiers

```text
Organization ID: almalgo_labs
Service ID: zk_llm1
Daemon ID: 2b4d4f3b-c043-4671-ad6a-697012038574
Hosted Service ID kept for rollback: cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
Coolify service URL: https://zkllm.almalgo.com
```

## Repository deployment files

Use `Dockerfile.fastapi` for the clearest Coolify deployment path.

The root `Dockerfile` is still valid for SNET Full-Stack HAAS by default. It
can also run FastAPI when `SERVICE_MODE=fastapi` or `PORT` is set, but
`Dockerfile.fastapi` avoids ambiguity for Coolify.

The Publisher proto remains:

```text
snet_service/proto/zk_llm_http_api.proto
```

It maps SNET methods onto HTTP routes:

```text
Health  -> POST /health
Session -> POST /api/session
Layer   -> POST /api/layer
```

## Coolify application settings

Use these settings:

```text
Deployment type: Dockerfile
Dockerfile path: Dockerfile.fastapi
Exposed port: 8000
Public URL: https://zkllm.almalgo.com
HTTPS: enabled
Healthcheck path: /heartbeat
WebSocket support: enabled if available
Restart policy: enabled
```

If you intentionally use the root `Dockerfile` in Coolify, set:

```bash
SERVICE_MODE=fastapi
PORT=8000
```

HTTP/2 or gRPC-specific proxy support is not required for this deployment,
because the service contract is HTTP passthrough.

Recommended persistent volume:

```text
/data/hf-cache
```

Recommended Coolify environment:

```bash
PORT=8000
ZKLLM_SERVER_MODEL_DTYPE=float32
HF_HOME=/data/hf-cache
TRANSFORMERS_CACHE=/data/hf-cache
```

Minimum resource target for real first-layer inference:

```text
RAM: 8 GB minimum, 16 GB preferred
CPU: 2 vCPU minimum, 4 vCPU preferred
Disk: enough for Python dependencies and HuggingFace model cache
```

## Public service contract

The Coolify service must return HTTP 200 for:

```text
GET  /
GET  /heartbeat
POST /heartbeat
GET  /health
POST /health
```

The HAAS daemon backend service endpoint must be the Coolify base URL only:

```text
https://zkllm.almalgo.com
```

Do not configure the daemon service endpoint with any path suffix:

```text
Do not use: https://zkllm.almalgo.com/health
Do not use: https://zkllm.almalgo.com/heartbeat
Do not use: https://zkllm.almalgo.com/api/session
Do not use: https://zkllm.almalgo.com/api/layer
```

## Publisher HAAS daemon setup

In Publisher:

1. Keep mainnet selected.
2. Do not create a new HAAS hosted service runtime.
3. Redeploy or update only the daemon.
4. Set the daemon backend service endpoint to the Coolify base URL.
5. Confirm the published metadata still maps to:
   - `almalgo_labs`
   - `zk_llm1`
   - the correct group and payment settings
   - the correct published daemon endpoint

If marketplace metadata still references the old full-stack hosted route,
republish metadata and allow indexing/cache propagation.

## Verification

Run local tests after code changes:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
pytest -q tests/test_customer_main.py server/tests/test_api_endpoint.py
```

Run local liveness checks:

```bash
uvicorn server.server:app --host 0.0.0.0 --port 8000
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/heartbeat
curl -i -X POST http://127.0.0.1:8000/heartbeat -H 'content-type: application/json' -d '{"op":"heartbeat"}'
curl -i http://127.0.0.1:8000/health
curl -i -X POST http://127.0.0.1:8000/health -H 'content-type: application/json' -d '{"op":"health"}'
```

Validate Coolify directly:

```bash
python3 scripts/snet_endpoint_preflight.py \
  --public-base-url "https://zkllm.almalgo.com"

python3 scripts/m5_snet_smoke.py \
  --base-url "https://zkllm.almalgo.com" \
  --output benchmarks/results/m5_snet_smoke_coolify_direct.json \
  --timeout 900
```

Validate the HAAS daemon after redeploy:

```bash
python3 scripts/snet_endpoint_preflight.py \
  --public-base-url "https://zkllm.almalgo.com" \
  --daemon-base-url "https://<mainnet-haas-daemon-domain>"

python3 scripts/m5_snet_smoke.py \
  --base-url "https://<mainnet-haas-daemon-domain>" \
  --output benchmarks/results/m5_snet_smoke_haas_daemon_coolify.json \
  --timeout 900
```

Run reliability evidence:

```bash
python3 scripts/m5_snet_reliability.py \
  --base-url "https://<mainnet-haas-daemon-domain>" \
  --attempts 20 \
  --concurrency 4 \
  --timeout 900 \
  --reliability-output benchmarks/results/m5_reliability_haas_daemon_coolify.json \
  --recovery-output benchmarks/results/m5_recovery_haas_daemon_coolify.json
```

## Rollback

Do not delete the existing HAAS hosted service during cutover.

Rollback options:

1. Repoint the daemon backend endpoint to the previous HAAS hosted service if Publisher allows it.
2. Redeploy the previous full-stack HAAS setup from the current root `Dockerfile`.
3. Preserve Coolify logs and HAAS daemon logs before changing configuration again.
4. Avoid republishing marketplace metadata unless the daemon endpoint itself changes.
