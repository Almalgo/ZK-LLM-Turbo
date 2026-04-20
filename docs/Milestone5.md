# Milestone 5: SingularityNet Integration Report

Date: 2026-04-14

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

Local preflight artifacts (backend endpoint validation):

- `benchmarks/results/m5_snet_smoke_local.json` (pass)
- `benchmarks/results/m5_reliability_local.json` (fail in this run due backend process interruption)
- `benchmarks/results/m5_recovery_local.json` (fail in this run; unrecovered failure streak observed)

Additional daemon-local probe artifact:

- `benchmarks/results/m5_snet_smoke_daemon_local.json` (current status: fail; daemon exited before serving target endpoint)

Target Milestone 5 artifacts (daemon/Sepolia evidence) pending runtime publication flow:

- `benchmarks/results/m5_snet_smoke_sepolia.json`
- `benchmarks/results/m5_reliability_sepolia.json`
- `benchmarks/results/m5_recovery_sepolia.json`

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

## Remaining Steps to Complete Milestone 5

1. Fill daemon runtime config with real values:
   - `ORG_ID`, `SERVICE_ID`, `DOMAIN`, `SEPOLIA_RPC_URL`, `MAINNET_RPC_URL`
2. Launch daemon with Sepolia config and run:
   - `scripts/m5_snet_smoke.py` against daemon endpoint
   - `scripts/m5_snet_reliability.py` against daemon endpoint
3. Save Sepolia evidence artifacts under `benchmarks/results/`.
4. Publish service on Mainnet and verify public link.
5. Update this report with:
   - final daemon endpoint/public service link
   - Sepolia and Mainnet verification evidence
   - final pass/fail status for milestone deliverables.

## Current Blockers

Mainnet publication and public service link verification are blocked pending operator-provided chain/runtime credentials and funded signer details.

Additional blocker observed during local daemon preflight:

- `snetd` v6.2.1 crashed with a nil-pointer panic when started in a no-blockchain local mode (`blockchain_enabled=false`) in this environment, preventing stable daemon-local passthrough verification.
- Crash evidence log: `benchmark-snetd-7021.log`.
- With `blockchain_enabled=true`, daemon startup failed fast until valid on-chain identifiers are provided (`organization_id`, `service_id`).
- Startup failure evidence log: `benchmark-snetd-7022.log`.
