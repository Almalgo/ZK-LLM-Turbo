# T3 Decision Criteria

Date: 2026-04-13

## Goal

Provide fast, evidence-driven criteria for whether current Phase 3 work supports a near-term change, without requiring a major architecture rewrite.

## Criteria

### T3.1 OpenFHE matmul readiness

- Accuracy requirement: OpenFHE matmul MAE <= `1e-5` on sampled benchmark dimensions
- Performance requirement: worst OpenFHE slowdown vs TenSEAL <= `5.0x`

Rationale:

- `1e-5` is loose enough for pragmatic HE numerical tolerance on this stage.
- `5.0x` is still strict for production switching, but realistic enough to identify actionable progress in short time.

### T3.3 GPU readiness

- `nvidia-smi` probe must be available
- OpenFHE readiness must already be `go`
- A usable GPU HE execution path must be demonstrable in this repo/runtime

### T3.4 polynomial readiness

- Server-side polynomial replacements must exist for:
  - RMSNorm
  - softmax

### T3.5 non-interactive readiness

- Requires both T3.3 and T3.4 readiness as `go`.

## Decision output

All decisions should be machine-readable under `benchmarks/results/` and summarized in:

- `docs/T3-PHASE3-READINESS-SUMMARY.md`
