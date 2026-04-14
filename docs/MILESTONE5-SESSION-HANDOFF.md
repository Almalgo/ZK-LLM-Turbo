# Milestone 5 Session Handoff

Date: 2026-04-14

## Goal

Deliver Milestone 5 (SingularityNet integration) with:

1. Public service link
2. Development report

Constraint: avoid major architecture rewrites and prioritize fast, evidence-driven progress.

## Current Status Before Milestone 5

Phase 3 evidence and decision pipeline is complete and published.

Final decision artifact:

- `benchmarks/results/t3_change_decision.json` -> `supports_change: false`

Supporting docs:

- `docs/T3-CHANGE-DECISION.md`
- `docs/T3-PHASE3-READINESS-SUMMARY.md`
- `docs/T3-DECISION-CRITERIA.md`

## Milestone 5 Architecture Decision

Chosen integration path:

- SingularityNet daemon HTTP passthrough to existing FastAPI backend

Rationale:

- Minimal code changes
- Fits timeline and current project constraints
- Avoids full gRPC wrapper rewrite for this milestone

## Relevant Existing API Surface

Primary backend endpoints:

- `POST /api/session`
- `POST /api/layer`

Legacy/back-compat endpoint:

- `POST /api/infer` (do not use as primary integration route)

## Milestone 5 Execution Plan

1. Create `snet_service/` integration scaffolding and daemon config templates (Sepolia + Mainnet)
2. Add smoke test script for daemon-routed session + one layer call
3. Add reliability/load artifacts (success rate, p50/p95, concurrency, recovery)
4. Rehearse full registration/publish flow on Sepolia
5. Publish on Mainnet and verify public link
6. Produce `docs/Milestone5.md` report with evidence links

## Inputs Needed During Execution

- `ORG_ID`
- `SERVICE_ID`
- `DOMAIN`
- `SEPOLIA_RPC_URL`
- `MAINNET_RPC_URL`
- Sepolia/Mainnet wallet/signer details
- Funding confirmation (gas/tokens)

## Initial Milestone 5 Artifacts to Produce

- `benchmarks/results/m5_snet_smoke_sepolia.json`
- `benchmarks/results/m5_reliability_sepolia.json`
- `benchmarks/results/m5_recovery_sepolia.json`
- `docs/Milestone5.md`

## Start Command Set (Next Session)

1. Implement Task 1:
   - `snet_service/snetd.config.sepolia.template.json`
   - `snet_service/snetd.config.mainnet.template.json`
   - `snet_service/README.md`
2. Implement Task 2:
   - `scripts/m5_snet_smoke.py`
3. Commit and push after each task.
