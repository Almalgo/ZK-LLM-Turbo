# Milestone 5: SingularityNET Integration Finalization Report

Date: 2026-04-26

## 1) Goal

Deliver Milestone 5 finalization for mainnet SingularityNET publishing using Hosting-as-a-Service (HaaS), and consolidate all implementation details, validation artifacts, and outstanding gaps into a completion report.

## 2) Architecture Decision

- Integration type: Hosting-as-a-Service (HaaS) for daemon management.
- Transport mode: HTTP passthrough from SNET daemon to existing FastAPI service.
- Primary endpoints:
  - `POST /api/session`
  - `POST /api/layer`
- Health/operational heartbeat endpoint:
  - `GET /heartbeat`
- Internal/auxiliary health endpoint retained:
  - `GET /health`

## 3) API Contract

- Service/method mapping (Publisher single proto requirement):
  - `snet_service/proto/zk_llm_http_api.proto` (single-file upload)
  - `ZKLLMService.Session` → `POST /api/session`
  - `ZKLLMService.Layer` → `POST /api/layer`
- Heartbeat contract:
  - `GET /heartbeat` must return HTTP 200 for runtime reachability checks.
- Payload fields used by the service:
  - `SessionRequest.public_context_b64`
  - `SessionResponse.session_id`
  - `LayerRequest.session_id`, `layer_idx`, `operation`, `encrypted_vectors_b64`
  - `LayerResponse.encrypted_results_b64`, `operation`, `layer_idx`, optional `elapsed_ms`

## 4) Delivered Implementation

Milestone 5 repository deliverables were implemented to support HaaS + HTTP passthrough:

- Added `server/server.py` heartbeat route:
  - `GET /heartbeat` returns `{ "status": "ok" }`.
- Updated `snet_service/snetd.config.mainnet.json`:
  - `organization_id`: `almalgo_labs`
  - `service_id`: `zk_llm1`
  - `daemon_type`: `http`
  - `service_endpoint` remains backend-targeted and is configurable per deployment.
- Added operational health guidance and single-proto upload constraints in `snet_service/README.md`.
- Updated docs status notes in `docs/Milestone5.md` to reflect service identifier and heartbeat-alignment state.
- Packaged and uploaded proto zip:
  - `/home/oussama/Downloads/zk_llm_http_api.zip` (single-file zip of `zk_llm_http_api.proto`).

## 5) Files/Artifacts Produced

- Core integration scaffolding:
  - `snet_service/snetd.config.mainnet.template.json`
  - `snet_service/snetd.config.mainnet.json`
  - `snet_service/proto/zk_llm_http_api.proto`
  - `snet_service/README.md`
  - `server/server.py` (`/heartbeat`)
- Validation scripts:
  - `scripts/m5_snet_smoke.py`
  - `scripts/m5_snet_reliability.py`
- Benchmark evidence:
  - `benchmarks/results/m5_snet_smoke_local.json`
  - `benchmarks/results/m5_reliability_local.json`
  - `benchmarks/results/m5_recovery_local.json`
  - `benchmarks/results/m5_snet_smoke_mainnet.json`
  - `benchmarks/results/m5_reliability_mainnet.json`
  - `benchmarks/results/m5_recovery_mainnet.json`
- Publisher proto package prepared:
  - `~/Downloads/zk_llm_http_api.zip`

## 6) Verification Commands Run

```bash
python3 scripts/m5_snet_smoke.py \
  --base-url "http://127.0.0.1:8011" \
  --timeout 300 \
  --output benchmarks/results/m5_snet_smoke_local.json

python3 scripts/m5_snet_reliability.py \
  --base-url "http://127.0.0.1:8012" \
  --attempts 3 \
  --concurrency 1 \
  --timeout 300 \
  --min-success-rate 1.0 \
  --reliability-output benchmarks/results/m5_reliability_local.json \
  --recovery-output benchmarks/results/m5_recovery_local.json
```

```bash
python3 scripts/m5_snet_smoke.py \
  --base-url "http://127.0.0.1:7000" \
  --output benchmarks/results/m5_snet_smoke_mainnet.json

python3 scripts/m5_snet_reliability.py \
  --base-url "http://127.0.0.1:8000" \
  --attempts 20 \
  --concurrency 4 \
  --reliability-output benchmarks/results/m5_reliability_mainnet.json \
  --recovery-output benchmarks/results/m5_recovery_mainnet.json
```

```bash
# Health checks used by runtime checks
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/heartbeat
```

## 7) Pass/Fail Evidence Matrix

| Check | Artifact | Expected | Observed | Status |
|---|---|---|---|---|
| Local smoke (`session` + `qkv` layer) | `benchmarks/results/m5_snet_smoke_local.json` | status=pass, `result_count = 3` | status=pass, `result_count = 3` | Pass |
| Local reliability (3 attempts, c=1) | `benchmarks/results/m5_reliability_local.json` | status=pass, `success_rate >= 1.0` | status=fail, `success_rate = 0.333333` | Fail |
| Local recovery (3 attempts, c=1) | `benchmarks/results/m5_recovery_local.json` | status=pass, `unrecovered_streaks = 0` | status=fail, `unrecovered_streaks = 1` | Fail |
| Mainnet smoke | `benchmarks/results/m5_snet_smoke_mainnet.json` | status=pass, `result_count = 3` | status=fail, `Session response missing 'session_id'` | Fail |
| Mainnet reliability (20 attempts, c=4) | `benchmarks/results/m5_reliability_mainnet.json` | status=pass, `success_rate >= 0.95` | status=fail, `success_rate = 0.35` | Fail |
| Mainnet recovery (20 attempts, c=4) | `benchmarks/results/m5_recovery_mainnet.json` | status=pass, `unrecovered_streaks = 0` | status=pass, `unrecovered_streaks = 0` | Pass |

### Mainnet exception note

- Current mainnet evidence reflects pre-portal-alignment/preflight conditions.
- Service metadata/config alignment has been corrected in repo (active service ID now `zk_llm1`; heartbeat endpoint added).
- Final public HaaS endpoint verification is still pending and these rows are expected to be overwritten after successful public endpoint smoke/reliability/recovery runs.

## 8) Current Blockers / Remaining Gaps

- Public endpoint verification not yet completed against final deployed HaaS route (service must be ONLINE in SNET portal).
- Signer/payment metadata completion in SNET publisher still required:
  - Free Call Signer Address
  - Metering Address
  - Mainnet-funded address readiness
- `domain` and `MAINNET_RPC_URL` values must remain aligned with final published daemon/runtime route.

## 9) Milestone 5 Publication Metadata

- Organization: `almalgo_labs`
- Service: `zk_llm1`
- Mode: Hosting-as-a-Service
- Mainnet portal link: TODO (publish and paste final service URL)
- Proto upload: `snet_service/proto/zk_llm_http_api.proto` uploaded as a single file zip (`zk_llm_http_api.zip`)
- Metering address + Free Call Signer: TODO (confirm funded wallet addresses in publisher)

## 10) Finalization Status

- Milestone 5 is **partially complete** (repo and validation scaffolding done, heartbeat/API contract aligned).
- Outstanding item: confirm final public mainnet service runs against live HaaS URL and refresh the evidence matrix to final-pass status.
