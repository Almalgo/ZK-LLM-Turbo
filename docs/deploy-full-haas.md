# SingularityNET Full-Stack HaaS Deployment

This is the preferred production deployment mode for ZK-LLM-Turbo when both the
daemon and the API/service runtime should be hosted by SingularityNET Publisher.

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
        v
customer_main.run(input_data)
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

The rest of the file contains the service runtime dependencies required by the
session and encrypted layer handlers.

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

For this mode, do not point the daemon to a Coolify URL. Publisher manages the
daemon and the service runtime.

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
{'serviceID': 'zk_llm1', 'status': 'SERVING'}
```

Run focused tests:

```bash
python3 -m pytest -q tests/test_customer_main.py
```

Build the root HaaS image locally:

```bash
docker build -f Dockerfile -t zk-llm-turbo-haas .
```

The image should start `runpod_handler.py`.

## Notes

- `Dockerfile.fastapi` remains available only for local/Coolify-style HTTP API
  testing.
- `docker-entrypoint.sh` is not used by the root HaaS image.
- Avoid loading model weights in `profile.json`; use the health operation for
  Publisher profiling.
