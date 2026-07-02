# SingularityNET Full-Stack HaaS Deployment

This is the preferred production deployment mode for ZK-LLM-Turbo when
SingularityNET Publisher should host the daemon and HaaS service object while
the heavy TinyLlama/TenSEAL runtime remains on self-hosted infrastructure.

## Architecture

```text
Marketplace / client
        |
        v
Publisher HAAS daemon
        |
        v
Publisher Full-Stack HAAS RunPod service
        |
        | customer_main.run(input_data)
        | forwards HTTP requests
        v
Self-hosted ZK-LLM FastAPI backend
```

The repository root is prepared for the Publisher Full-Stack HaaS contract:

```text
customer_main.py
requirements.txt
profile.json
runpod_handler.py
Dockerfile
```

The root `Dockerfile` starts:

```text
python -u runpod_handler.py
```

`runpod_handler.py` imports `customer_main.py` and calls:

```python
run(input_data)
```

This matches the current HaaS repository contract documented by
SingularityNET: Full-Stack mode builds a Docker container from the GitHub repo,
deploys it as a serverless endpoint, and passes `profile.json` input directly
to `customer_main.run`.

The HaaS service is intentionally a lightweight proxy. It does not load
TinyLlama or perform encrypted layer inference locally.

## Service operations

`customer_main.run(input_data)` supports:

```text
op=health
op=heartbeat
op=session
op=layer
```

Layer operations may also be routed by operation name:

```text
qkv
o_proj
ffn_gate_up
ffn_down
ffn_merged
```

Health and profiling are intentionally lightweight:

```json
{
  "serviceID": "zk_llm1",
  "status": "SERVING"
}
```

This avoids loading TinyLlama during the Publisher profiling step.

## Proxy environment

Configure these environment variables in the Publisher Full-Stack HaaS service:

```env
ZKLLM_BACKEND_BASE_URL=https://zkllm.almalgo.com
ZKLLM_BACKEND_AUTH_TOKEN=
ZKLLM_BACKEND_TIMEOUT_SECONDS=900
ZKLLM_BACKEND_HEALTH_TIMEOUT_SECONDS=10
ZKLLM_PROXY_FAIL_OPEN_HEALTH=true
```

Behavior:

- `ZKLLM_BACKEND_BASE_URL` is required for `session` and `layer` operations.
- If `ZKLLM_BACKEND_AUTH_TOKEN` is set, the proxy forwards it as a bearer token.
- Health and profiling stay `SERVING` by default even if the backend is down.
- Backend reachability diagnostics are included in the health response.

Example health response:

```json
{
  "serviceID": "zk_llm1",
  "service": "zk-llm-turbo",
  "status": "SERVING",
  "mode": "proxy",
  "backend": {
    "configured": true,
    "reachable": true,
    "base_url": "https://zkllm.almalgo.com",
    "status_code": 200,
    "error": null
  }
}
```

## Backend endpoint contract

The self-hosted backend must expose:

```text
GET  /health
POST /health
POST /api/session
POST /api/layer
```

The HaaS proxy forwards:

```text
session -> POST /api/session
layer   -> POST /api/layer
```

## Profile payload

`profile.json` should stay lightweight:

```json
{
  "input": {
    "op": "health"
  }
}
```

The platform sends this object to `customer_main.run(input_data)` during
profiling. If this fails, the hosted service status moves to `ERROR`.

## Root dependencies

Keep these platform dependencies in `requirements.txt`:

```text
runpod==1.7.12
sentry-sdk==2.46.0
```

The rest of the file contains the lightweight proxy dependencies and optional
local FastAPI/backend dependencies used by development and rollback paths.

## Publisher setup

Use Full-Stack HaaS in Publisher:

1. Connect the GitHub repository default branch.
2. Confirm the repository root contains the required files.
3. Use the existing service metadata/proto:
   - `snet_service/proto/zk_llm_http_api.proto`
   - package/service: `zk_llm.ZKLLMService`
4. Deploy the hosted service and daemon through Publisher.
5. Watch the lifecycle:

```text
VALIDATING -> REGISTERING -> PUSHING_NEW_VERSION -> BUILDING -> DEPLOYING -> PROFILING -> UP
```

For this mode, do not configure the daemon-only service endpoint directly to
Coolify. Publisher manages the daemon and calls the Publisher-hosted HaaS proxy,
which then forwards to the self-hosted backend.

## Local verification

Run the lightweight HaaS entrypoint check:

```bash
python3 - <<'PY'
from customer_main import run
print(run({"op": "health"}))
PY
```

Expected output:

```text
{'serviceID': 'zk_llm1', 'service': 'zk-llm-turbo', 'status': 'SERVING', 'mode': 'proxy', ...}
```

Run focused tests:

```bash
python3 -m pytest -q tests/test_customer_main.py
```

Run proxy health against a real backend:

```bash
ZKLLM_BACKEND_BASE_URL=https://zkllm.almalgo.com \
python3 - <<'PY'
from customer_main import run
print(run({"op": "health"}))
PY
```

Build the root HaaS image locally:

```bash
docker build -f Dockerfile -t zk-llm-turbo-haas .
```

The image should start `runpod_handler.py`.

## Notes

- `Dockerfile.fastapi` remains available only for local/Coolify-style HTTP API
  testing.
- Avoid loading model weights in `profile.json`; use the health operation for
  Publisher profiling.
