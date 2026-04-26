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

## Hosting-as-a-Service (HaaS) in Publisher (recommended)

If you chose **Hosting-as-a-Service (HaaS)** in the Publisher portal, you do not need local `etcd` setup and you normally do not run `snetd` locally.

For HaaS, your repository only needs to expose the AI service API that the hosted daemon will call:

- `POST /api/session`
- `POST /api/layer`

Minimum publish requirements:

- Public and reachable service endpoint URL (HTTP/HTTPS)
- Optional request authentication (if required by your deployment)
- Stable backend process with these routes available

Operational difference:

- Self-hosted: you manage daemon + service + SSL + `etcd` + scaling
- HaaS: SNET manages daemon, SSL, and managed `etcd` for you

You can still use local `snet_service/*` configs for preflight smoke/reliability checks; they are optional when publishing with HaaS.

## API contract for Publisher (HTTP + 1 proto file)

For SNET Hosting-as-a-Service HTTP service registration, use one proto file:

- `snet_service/proto/zk_llm_http_api.proto`

Required methods and mapping:

- `Session` -> `POST /api/session`
  - request: `public_context_b64`
  - response: `session_id`
- `Layer` -> `POST /api/layer`
  - request: `session_id`, `layer_idx`, `operation`, `encrypted_vectors_b64`
  - response: `encrypted_results_b64`, `operation`, `layer_idx`, optional `elapsed_ms`

Operational health check contract:

- `GET /heartbeat` should return HTTP 200 for SNET heartbeat checks.
- `GET /health` is also kept for internal monitoring parity.

Upload exactly this single proto file in Publisher (HTTP services are one-proto limited).

## Hosted dashboard action controls: why they can disappear

When the service status is not `UP` (for example `OFFLINE`, `UNHEALTHY`, or still initializing), the portal may hide service controls and keep Logs unavailable.

- This is expected if heartbeat is failing.
- You typically need at least a successful external heartbeat probe and visible endpoint health before actions become available.

Recommended operator checks while in dashboard read-only state:

1. Confirm the service is selected as:
   - Organization: `almalgo_labs`
   - Service: `zk_llm1`
   - Hosting-as-a-Service enabled
2. Confirm the daemon endpoint is the public backend base URL, not the HaaS orchestrator API URL.
3. Confirm both routes return 200:
   - `GET /health`
   - `GET /heartbeat`
4. Run public smoke/reliability checks against the same URL before returning to the portal:
   - `python3 scripts/m5_snet_smoke.py --base-url "<PUBLIC_BASE_URL>" --output benchmarks/results/m5_snet_smoke_mainnet.json`
   - `python3 scripts/m5_snet_reliability.py --base-url "<PUBLIC_BASE_URL>" --attempts 20 --concurrency 4 --reliability-output benchmarks/results/m5_reliability_mainnet.json --recovery-output benchmarks/results/m5_recovery_mainnet.json`
5. Reopen the service page after propagation.
   - Logs and action controls should appear once status becomes `UP`.

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
