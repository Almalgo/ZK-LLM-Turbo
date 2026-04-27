# ZK-LLM-Turbo Agent Context

## Active Workstream

Milestone 5: SingularityNet integration.

Deliverables:

1. Public service link
2. Development report

## Execution Constraints

- Prefer targeted, minimal-intrusion changes.
- Avoid major architectural rewrites.
- Use evidence artifacts to drive support/no-support decisions.
- Commit incrementally and push after each completed task.

## Milestone 5 Integration Decision

- Use SingularityNet daemon HTTP passthrough as primary path.
- Keep FastAPI backend as-is where possible.
- Primary API integration should target:
  - `POST /api/session`
  - `POST /api/layer`

## Key Existing Evidence (Phase 3)

- `benchmarks/results/t3_change_decision.json` (final support/no-support decision)
- `benchmarks/results/t3_phase3_gate.json`

## Milestone 5 Working Plan

1. Create `snet_service/` scaffolding and daemon config templates (Sepolia + Mainnet)
2. Add daemon-path smoke script and artifact output
3. Add reliability/load test harness and artifacts
4. Sepolia registration/publish rehearsal
5. Mainnet publication and public link verification
6. Final development report in `Reports/Milestone5.md`

## Inputs Required from Operator

- ORG_ID
- SERVICE_ID
- DOMAIN
- SEPOLIA_RPC_URL
- MAINNET_RPC_URL
- Wallet/signer and funding details for Sepolia + Mainnet

## Target Milestone 5 Artifacts

- `benchmarks/results/m5_snet_smoke_sepolia.json`
- `benchmarks/results/m5_reliability_sepolia.json`
- `benchmarks/results/m5_recovery_sepolia.json`
- `Reports/Milestone5.md`

## Current Handoff Document

- `Reports/Milestone5.md`
