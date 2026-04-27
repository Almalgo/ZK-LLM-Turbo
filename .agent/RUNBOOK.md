# Milestone 5 Runbook

## Objective

Validate SingularityNet daemon passthrough for primary backend API route:

- `POST /api/session`
- `POST /api/layer`

## Preflight (code-level)

```bash
source split-inference-env/bin/activate
python -m py_compile scripts/m5_snet_smoke.py scripts/m5_snet_reliability.py
python -m json.tool snet_service/snetd.config.sepolia.template.json >/dev/null
python -m json.tool snet_service/snetd.config.mainnet.template.json >/dev/null
```

## Sepolia Execution

1) Start backend and daemon using real values.

2) Smoke artifact:

```bash
python scripts/m5_snet_smoke.py --base-url "http://<daemon-host>:7000" --output benchmarks/results/m5_snet_smoke_sepolia.json
```

3) Reliability/recovery artifacts:

```bash
python scripts/m5_snet_reliability.py \
  --base-url "http://<daemon-host>:7000" \
  --attempts 20 \
  --concurrency 4 \
  --reliability-output benchmarks/results/m5_reliability_sepolia.json \
  --recovery-output benchmarks/results/m5_recovery_sepolia.json
```

## Mainnet Completion

1) Use mainnet template with real values.
2) Publish/update service and verify publicly reachable service URL.
3) Record final link and outcome in `Reports/Milestone5.md`.

## Expected Deliverables

- `benchmarks/results/m5_snet_smoke_sepolia.json`
- `benchmarks/results/m5_reliability_sepolia.json`
- `benchmarks/results/m5_recovery_sepolia.json`
- `Reports/Milestone5.md`
