# Milestone 5: SingularityNet Integration Report

Date: 2026-04-25

## Goal

Deliver Milestone 5 integration with:

1. Public service link
2. Development report

Approach: daemon HTTP passthrough to existing FastAPI backend, preserving the current split-inference protocol surface.

## Architecture Decision

Chosen path:

- SingularityNet daemon passthrough to existing backend endpoints:
  - `POST /api/session`
  - `POST /api/layer`

Legacy endpoint status:

- `POST /api/infer` remains backward-compat only and is not the primary Milestone 5 route.

### API contract for Hosting-as-a-Service

- Operational health endpoint:
  - `GET /heartbeat`

- API method set must be represented in a **single proto file**.
- Created for this run:
  - `snet_service/proto/zk_llm_http_api.proto`
- Service/methods expected by Publisher:
  - `ZKLLMService`
  - `Session` -> `POST /api/session`
  - `Layer` -> `POST /api/layer`
- Request/response field names aligned with backend handlers:
  - `public_context_b64`
  - `session_id`
  - `layer_idx`
  - `operation`
  - `encrypted_vectors_b64`
  - `encrypted_results_b64`
  - `elapsed_ms` (optional)

## Delivered Implementation

### Task 1: daemon scaffolding

Added:

- `snet_service/snetd.config.sepolia.template.json`
- `snet_service/snetd.config.mainnet.template.json`
- `snet_service/README.md`

Highlights:

- Placeholder-based templates for `ORG_ID`, `SERVICE_ID`, `DOMAIN`, and chain RPC URLs.
- Passthrough-first config with `service_endpoint` targeting the existing FastAPI backend.
- Operator guide for local config materialization and daemon execution.

### Task 2: daemon-path smoke harness

Added:

- `scripts/m5_snet_smoke.py`

Capabilities:

- Creates CKKS context and serializes public context.
- Calls session setup endpoint.
- Executes one encrypted `qkv` layer request.
- Validates response structure (`3` encrypted outputs expected for `qkv`).
- Writes machine-readable artifact on both pass and fail outcomes.

### Reliability/recovery harness

Added:

- `scripts/m5_snet_reliability.py`

Capabilities:

- Runs repeated session+layer attempts with configurable concurrency.
- Reports reliability metrics (success rate, p50/p95 latencies).
- Derives recovery metrics from observed failure streaks and subsequent recoveries.
- Writes separate reliability and recovery artifacts.

## Artifacts Produced in This Session

Mainnet config artifact (prepared locally):

- `snet_service/snetd.config.mainnet.json`
  - `organization_id`: `almalgo_labs`
  - `service_id`: `zk_llm1`
  - `auto_ssl_domain`: `localhost` (for local smoke/running), production requires routable `DOMAIN`
  - `ethereum_json_rpc_http_endpoint`: `https://cloudflare-eth.com`

Local preflight artifacts (backend endpoint validation):

- `benchmarks/results/m5_snet_smoke_local.json` (pass)
- `benchmarks/results/m5_reliability_local.json` (fail in this run due backend process interruption)
- `benchmarks/results/m5_recovery_local.json` (fail in this run; unrecovered failure streak observed)

Additional daemon-local probe artifact:

- `benchmarks/results/m5_snet_smoke_daemon_local.json` (current status: fail; daemon exited before serving target endpoint)

Target Milestone 5 artifacts (mainnet evidence) pending runtime publication flow:

- `benchmarks/results/m5_snet_smoke_mainnet.json`
- `benchmarks/results/m5_reliability_mainnet.json`
- `benchmarks/results/m5_recovery_mainnet.json`

## Commands Used

Preflight smoke (local backend route):

```bash
./split-inference-env/bin/python scripts/m5_snet_smoke.py \
  --base-url "http://127.0.0.1:8011" \
  --timeout 300 \
  --output benchmarks/results/m5_snet_smoke_local.json
```

Preflight reliability/recovery:

```bash
./split-inference-env/bin/python scripts/m5_snet_reliability.py \
  --base-url "http://127.0.0.1:8012" \
  --attempts 3 \
  --concurrency 1 \
  --timeout 300 \
  --min-success-rate 1.0 \
  --reliability-output benchmarks/results/m5_reliability_local.json \
  --recovery-output benchmarks/results/m5_recovery_local.json
```

## Remaining Steps to Complete Milestone 5 (Mainnet Fast Path)

1. Provide real values:
   - `DOMAIN`
   - `MAINNET_RPC_URL`
2. Fill those values in `snet_service/snetd.config.mainnet.json`.
3. Launch backend + `snetd` with mainnet config (or switch to Publisher HaaS flow where SNET hosts the daemon).
4. Run:
   - `python scripts/m5_snet_smoke.py --base-url "http://127.0.0.1:7000" --output benchmarks/results/m5_snet_smoke_mainnet.json`
   - `python scripts/m5_snet_reliability.py --base-url "http://127.0.0.1:7000" --attempts 20 --concurrency 4 --reliability-output benchmarks/results/m5_reliability_mainnet.json --recovery-output benchmarks/results/m5_recovery_mainnet.json`
5. Save mainnet evidence artifacts under `benchmarks/results/`.
6. Publish service on Mainnet and verify public link.
   - If using HaaS, confirm local etcd endpoint is not required.
7. Confirm API upload is exactly one file:
   - `snet_service/proto/zk_llm_http_api.proto`
8. Update this report with:
   - final daemon endpoint/public service link
   - mainnet verification evidence
   - pass/fail status for milestone deliverables.

## Verified Runtime Status (2026-04-25)

Tooling install status:
- Installed `snetd` to `~/.local/bin/snetd` (v6.2.1) and confirmed binary availability.

Mainnet daemon execution status:
- Starting `snetd` with `snet_service/snetd.config.mainnet.json` and `almalgo_labs`/`zk_llm1` fails at startup:
  - `error retrieving contract details for the given organization and service ids Internal error`.
- Non-chain debug attempt (`"blockchain_enabled": false`, `auto_ssl_domain: ""`) allows daemon to start but smoke probe fails:
  - `Session response missing 'session_id'` in `benchmarks/results/m5_snet_smoke_mainnet.json`.
  - This indicates non-chain mode is not producing mainnet-usable session handoff behavior.

## Current Blockers

Mainnet publication and public service link verification are currently blocked pending:

- Runtime environment setup (`python`/`pip` and dependencies) *(completed)*
- `snetd` binary availability *(completed)*
- Confirmed mainnet org/service registration and metadata visibility for `almalgo_labs` + `zk_llm1`.
- Operator-provided runtime values (`DOMAIN`, `MAINNET_RPC_URL`)
- Funded signer details

Additional blocker observed during local daemon preflight:

- `snetd` v6.2.1 can panic with nil-pointer on shutdown in this non-production local run pattern.
- `blockchain_enabled=true` startup currently blocks on unresolved mainnet org/service contract details.
