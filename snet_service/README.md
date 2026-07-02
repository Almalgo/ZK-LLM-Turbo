# SingularityNet Daemon Scaffolding (Milestone 5)

This directory contains starter templates for routing SingularityNet daemon traffic to the existing FastAPI backend.

## Purpose

- Keep backend architecture unchanged.
- Use daemon HTTP passthrough as the Milestone 5 integration path.
- Templates set `daemon_type` to `http` for passthrough-style endpoint testing.
- Validate primarily through:
  - `POST /api/session`
  - `POST /api/layer`

The legacy `POST /api/infer` endpoint is backward-compat only and not the primary Milestone 5 route.

## Preferred mainnet deployment: Full-Stack HaaS

The preferred mainnet deployment is SingularityNET Publisher Full-Stack HaaS:

- Publisher runs the daemon.
- Publisher builds and runs a lightweight proxy service from the repository root.
- The proxy forwards session/layer work to the self-hosted FastAPI backend.
- The root `Dockerfile` starts `python -u runpod_handler.py`.
- `runpod_handler.py` calls `customer_main.run(input_data)`.

The required root files are:

```text
customer_main.py
requirements.txt
profile.json
runpod_handler.py
Dockerfile
```

Detailed operator instructions are in:

```text
docs/deploy-full-haas.md
```

## Full-Stack Hosting-as-a-Service (HaaS) in Publisher

If you chose **Full-Stack Hosting-as-a-Service (HaaS)** in the Publisher portal, SNET builds and deploys the service from the GitHub repository and separately manages the daemon.

Official HaaS repository requirements are strict. The repository root must contain:

```text
customer_main.py
requirements.txt
profile.json
runpod_handler.py
Dockerfile
```

For Full-Stack HaaS, the root `Dockerfile` is the SNET/RunPod handler image, not the FastAPI server image. The root handler starts:

```text
python -u runpod_handler.py
```

`Dockerfile.fastapi` remains available for optional local/Coolify-style HTTP testing.

`runpod_handler.py` imports `customer_main.py` and calls:

```python
run(input_data)
```

The lightweight deployment profiling request is configured in `profile.json`:

```json
{
  "input": {
    "op": "health"
  }
}
```

This validates the handler without loading TinyLlama or requiring encrypted inference fixtures.

Supported HaaS proxy operations:

- `health`: returns local serving status plus backend diagnostics without model loading.
- `session`: forwards the existing `POST /api/session` payload to the configured backend.
- `layer`: forwards the existing `POST /api/layer` payload to the configured backend.

Required proxy environment:

```env
ZKLLM_BACKEND_BASE_URL=https://zkllm.almalgo.com
ZKLLM_BACKEND_AUTH_TOKEN=
ZKLLM_BACKEND_TIMEOUT_SECONDS=900
ZKLLM_BACKEND_HEALTH_TIMEOUT_SECONDS=10
ZKLLM_PROXY_FAIL_OPEN_HEALTH=true
```

SNET Publisher watches the GitHub repository default branch. Push HaaS deployment changes to the default branch, or change the repository default branch before expecting Publisher to redeploy.

Expected Hosted Service lifecycle:

```text
VALIDATING -> REGISTERING -> PUSHING_NEW_VERSION -> BUILDING -> DEPLOYING -> PROFILING -> UP
```

If the Hosted Service reaches `ERROR`, inspect **Hosted Service** logs/events, not daemon logs first. Daemon logs usually only show that the hosted service heartbeat is offline.

## FastAPI Docker mode

The FastAPI app remains available for local development, local daemon passthrough tests, and external HTTP hosting. Use `Dockerfile.fastapi` for this mode:

```bash
docker buildx build --load -f Dockerfile.fastapi -t zk-llm-turbo-fastapi .
docker run --rm -p 8000:8000 zk-llm-turbo-fastapi
```

When running the FastAPI image, the service must answer liveness probes before model loading:

- `GET /`
- `GET /heartbeat`
- `GET /health`

For FastAPI passthrough deployments, the Daemon **Service Endpoint** must point to the reachable FastAPI service base URL.

Do not use these as the Daemon Service Endpoint:

- A GitHub repository URL
- `https://api.haas.singularitynet.dev/orchestrator/...`
- `http://127.0.0.1:8000` or `http://localhost:8000`
- Any URL with `/api/session`, `/api/layer`, `/health`, or `/heartbeat` appended

The FastAPI app must expose the API routes that the daemon will call:

- `POST /api/session`
- `POST /api/layer`

Minimum FastAPI service requirements:

- `Dockerfile.fastapi` builds successfully.
- Container starts with `uvicorn server.server:app --host 0.0.0.0 --port ${PORT:-8000}`.
- `GET /`, `GET /heartbeat`, and `GET /health` return HTTP 200 without loading TinyLlama.
- `POST /api/session` and `POST /api/layer` are available for daemon-forwarded calls.

Operational difference:

- Self-hosted: you manage daemon + service + SSL + `etcd` + scaling
- Full-Stack HaaS: SNET builds the root RunPod-style handler image and manages the daemon/serverless infrastructure
- FastAPI Docker mode: you run the HTTP backend yourself or use it for local daemon passthrough tests

You can still use local `snet_service/*` configs for FastAPI preflight smoke/reliability checks; they are optional when publishing with Full-Stack HaaS.

## API contract for Publisher (HTTP + 1 proto file)

For SNET Hosting-as-a-Service HTTP service registration, use one proto file:

- `snet_service/proto/zk_llm_http_api.proto`

Required methods and mapping:

- `Health` -> `POST /health`
  - request: `op` (set to `health` for Marketplace demo calls)
  - response: `status`, `service`, `model`, `model_status`, optional `model_error`
  - note: the FastAPI app also supports `GET /health` for Coolify/internal probes
- `Session` -> `POST /api/session`
  - request: `public_context_b64`
  - response: `session_id`
- `Layer` -> `POST /api/layer`
  - request: `session_id`, `layer_idx`, `operation`, `encrypted_vectors_b64`
  - response: `encrypted_results_b64`, `operation`, `layer_idx`, optional `elapsed_ms`

Operational health check contract:

- `GET /` should return HTTP 200 for default hosted-service probes.
- `GET /heartbeat` should return HTTP 200 for SNET heartbeat checks.
- `GET /health` is also kept for internal monitoring parity.

Upload exactly this single proto file in Publisher (HTTP services are one-proto limited).

## Marketplace demo UI

The Publisher demo archive is tracked at:

- `demo/zk-llm-turbo-marketplace-demo.zip`

The source files are in:

- `demo/marketplace-ui/`

The demo calls the lightweight `Health` RPC so users can verify the live Hosted
Service without generating CKKS keys or encrypted layer payloads in the browser.

## Hosted dashboard action controls: why they can disappear

When the service status is not `UP` (for example `OFFLINE`, `UNHEALTHY`, or still initializing), the portal may hide service controls and keep Logs unavailable.

- This is expected if heartbeat is failing.
- You typically need at least a successful external heartbeat probe and visible endpoint health before actions become available.

Recommended operator checks while in dashboard read-only state:

1. Confirm the service is selected as:
   - Organization: `almalgo_labs`
   - Service: `zk_llm1`
   - Hosting-as-a-Service enabled
2. Confirm the Hosted Service status is `UP`; if it is `ERROR`, inspect Hosted Service logs before daemon logs.
3. Confirm the Daemon Service Endpoint is the SNET Hosted Service base URL, not GitHub, localhost, or the HaaS orchestrator API URL.
4. Confirm these routes return 200 on the Hosted Service URL:
   - `GET /`
   - `GET /health`
   - `GET /heartbeat`
5. Run endpoint preflight and smoke/reliability checks against the same URL before returning to the portal:
   - `python3 scripts/snet_endpoint_preflight.py --public-base-url "<HOSTED_SERVICE_BASE_URL>"`
   - `python3 scripts/m5_snet_smoke.py --base-url "<HOSTED_SERVICE_BASE_URL>" --output benchmarks/results/m5_snet_smoke_hosted_service.json`
   - `python3 scripts/m5_snet_reliability.py --base-url "<HOSTED_SERVICE_BASE_URL>" --attempts 20 --concurrency 4 --reliability-output benchmarks/results/m5_reliability_hosted_service.json --recovery-output benchmarks/results/m5_recovery_hosted_service.json`
6. Reopen the service page after propagation.
   - Logs and action controls should appear once status becomes `UP`.

If HaaS daemon logs show heartbeat checks against
`https://api.haas.singularitynet.dev/orchestrator/...` returning 503, first make
the Hosted Service `UP`, then update the Daemon Service Endpoint to the Hosted
Service base URL and redeploy/restart the daemon. After the hosted service returns
HTTP 200 for `/`, `/heartbeat`, and `/health`, reset marketplace health backoff:

```text
https://marketplace-mt-v2.singularitynet.io/service-status/org/almalgo_labs/service/zk_llm1/health/reset
```

## Files

- `snetd.config.sepolia.template.json`
- `snetd.config.mainnet.template.json`
- `snetd.config.local.template.json`
- `proto/zk_llm_http_api.proto` (single-file HTTP API definition for Publisher)

## Required Operator Inputs

- `ORG_ID`
- `SERVICE_ID`
- `DOMAIN`
- `SEPOLIA_RPC_URL`
- `MAINNET_RPC_URL`
- Wallet/signer details and funding for Sepolia/Mainnet

## Quick Start

1. Copy a template to a local config file:

   - `cp snet_service/snetd.config.sepolia.template.json snet_service/snetd.config.sepolia.json`
   - `cp snet_service/snetd.config.mainnet.template.json snet_service/snetd.config.mainnet.json`

2. Replace placeholder values (`<ORG_ID>`, `<SERVICE_ID>`, `<DOMAIN>`, RPC URL placeholders).

3. Ensure `service_endpoint` points at your reachable backend, for example:

   - `http://127.0.0.1:8000` (same host)
   - `http://host.docker.internal:8000` (containerized daemon)
   - `http://<private-backend-host>:8000` (remote/private network)

4. Run daemon with your chosen config:

  - `snetd -c snet_service/snetd.config.sepolia.json`

5. Run smoke validation against daemon endpoint:

   - `python3 scripts/m5_snet_smoke.py --base-url "http://<daemon-host>:7000"`

6. Run reliability/recovery validation (recommended):

   - `python3 scripts/m5_snet_reliability.py --base-url "http://<daemon-host>:7000"`

## Optional local daemon preflight (no blockchain)

Use this when you want to validate passthrough wiring before Sepolia/Mainnet publication.

1. Copy local template:

   - `cp snet_service/snetd.config.local.template.json snet_service/snetd.config.local.json`

2. Point `service_endpoint` to your running FastAPI server.

3. Run daemon with local config:

   - `snetd -c snet_service/snetd.config.local.json`

4. Run smoke/reliability against daemon URL as above.

## Notes

- Keep filled config files with secrets out of version control.
- These templates are intentionally minimal and should be extended only as needed for your deployment model.
- Some daemon builds may warn that `daemon_type=http` is not intended for production defaults; this is still the selected Milestone 5 passthrough route.
- In this environment, `snetd` v6.2.1 crashed in `blockchain_enabled=false` local mode during preflight; use Sepolia/Mainnet configs with real `organization_id`/`service_id` for authoritative validation.
