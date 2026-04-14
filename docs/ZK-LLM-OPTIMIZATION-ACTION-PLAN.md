# ZK-LLM-Turbo Performance Optimization Action Plan

**Date:** 2026-03-29
**Team Size:** 2-3 engineers
**Scope:** Take per-token latency from 3-8s (1 layer) to <1s (22 layers) across 4 phases
**Based on:** [Performance Research Report](ZK-LLM-PERFORMANCE-RESEARCH.md)

---

## Table of Contents

1. [Current State Summary](#1-current-state-summary)
2. [Already Implemented](#2-already-implemented)
3. [Research Doc Accuracy Corrections](#3-research-doc-accuracy-corrections)
4. [Dependency Graph](#4-dependency-graph)
5. [Phase 1: Foundation & Quick Wins (Weeks 1-2)](#5-phase-1-foundation--quick-wins-weeks-1-2)
6. [Phase 2: Algorithmic Improvements (Weeks 3-6)](#6-phase-2-algorithmic-improvements-weeks-3-6)
7. [Phase 3: Architecture Migration (Months 2-5)](#7-phase-3-architecture-migration-months-2-5)
8. [Phase 4: Research Frontier (Months 6-12)](#8-phase-4-research-frontier-months-6-12)
9. [Performance Projections](#9-performance-projections)
10. [Verification Strategy](#10-verification-strategy)
11. [Critical Files Reference](#11-critical-files-reference)

---

## 1. Current State Summary

ZK-LLM-Turbo implements privacy-preserving split inference on TinyLlama 1.1B using CKKS homomorphic encryption (TenSEAL). The codebase is ~1,800 lines of Python across client, server, and common modules.

**Architecture:** 4 HTTP round-trips per transformer layer (QKV -> RoPE+Attention -> O -> Gate+Up -> SiLU -> Down)

**CKKS Parameters:**
- `poly_modulus_degree`: 8192 (4096 SIMD slots)
- `coeff_mod_bit_sizes`: [60, 40, 40, 60] (multiplicative depth ~2)
- `global_scale`: 2^40

**Current Performance:**
| Encrypted Layers | Round-trips/token | Est. Time/token | Practical? |
|---|---|---|---|
| 1/22 | 4 | 3-8s | Barely |
| 5/22 | 20 | 15-40s | No |
| 22/22 | 88 | 70-180s | Absolutely not |

**Plaintext TinyLlama:** <50ms/token

---

## 2. Already Implemented

The following optimizations from the research doc are **already shipped**. DO NOT re-implement:

| Optimization | Location | Details |
|---|---|---|
| HTTP Session reuse | `client/inference/layer_protocol.py:48-70` | `requests.Session()` with persistent auth headers |
| Weight `.tolist()` caching | `server/model/weight_manager.py:68-95` | Cached per layer, pre-split for >SLOT_COUNT matrices |
| Binary msgpack transport | `server/handlers/inference_handler.py:146-247` | `/api/layer/binary` endpoint with msgpack encoding |
| Zstd ciphertext compression | `layer_protocol.py:79-85`, `inference_handler.py:226` | Compression on both client and server |
| Server-side parallel QKV | `server/inference/he_ops.py:107-137` | `ThreadPoolExecutor(4)` with serialize/deserialize copy pattern |
| Server-side parallel FFN | `server/inference/he_ops.py:149-182` | Thread pool for parallel chunk matmuls |
| AVX512/HEXL detection | `server/server.py:15-25` | Logs warning if unavailable (detection only, not enabled) |
| Defense-in-depth limits | `inference_handler.py:152-158` | 10MB compressed, 50MB decompressed payload caps |

---

## 3. Research Doc Accuracy Corrections

The following estimates from the research doc are **overstated or inaccurate** given the actual codebase:

| Claim | Research Doc Estimate | Corrected Estimate | Rationale |
|---|---|---|---|
| Selective Galois Keys | 50-70% key size reduction | ~15% at N=8192 | Full set: 13 keys, needed: 11. 50-70% applies to N=16384+ |
| Scale factor reduction | 1.2-1.5x speedup | 1.05-1.15x | NTT cost dominates; driven by poly degree, not scale |
| Diagonal packing | 10-30x matmul speedup | 5-15x end-to-end | Rotation savings real, but encoding/accumulation overhead reduces net gain |
| GPU acceleration | 100-200x | 20-50x for leveled ops | 100-200x applies to bootstrapping-heavy workloads; our current leveled approach sees less benefit |
| Weight pre-conversion | Listed as TODO | Already implemented | `weight_manager.py:68-95` caches `.tolist()` at load time |
| Connection pooling | Listed as "5 minute fix" | Already implemented | `layer_protocol.py` uses `requests.Session()` |
| Binary transport | Listed as TODO | Already implemented | `/api/layer/binary` endpoint with msgpack+zstd |

---

## 4. Dependency Graph

```
Phase 1 (Weeks 1-2)
  T1.1  Benchmark Infrastructure ──────────────────────────────────┐
  T1.2  Session Cleanup ───────────────────────────────────────────┤
  T1.3  Multi-Token Packing ───────────────────────────────────────┤
  T1.4  Selective Galois Keys (investigate only) ──────────────────┤
  T1.5  CI Skeleton + CLAUDE.md ───────────────────────────────────┤
  T1.6  Scale Factor Experiment ───────────────────────────────────┤
                                                                    │
Phase 2 (Weeks 3-6)                                                 │
  T2.1  Intel HEXL Enablement ────────── requires T1.1 ───────────┤
  T2.2  Polynomial SiLU Approx ──────── requires T1.1, T1.6 ─────┤
  T2.3  Deeper CKKS Params (N=16384) ── requires T2.2 ────────────┤
  T2.4  WebSocket Connection ─────────── requires T1.1 ────────────┤
  T2.5  Async Pipelining ────────────── requires T2.4 ────────────┤
  T2.6  Phase 2 Accuracy Gate ───────── requires T2.2, T2.3 ──────┤
                                                                    │
Phase 3 (Months 2-5)                                                │
  T3.1  TenSEAL -> OpenFHE Migration ── requires T2.1, T2.3 ─────┤
  T3.2  Diagonal Packing MatMul ──────── requires T3.1 ───────────┤
  T3.3  GPU-Accelerated CKKS ─────────── requires T3.1 ───────────┤
  T3.4  Full Polynomial Model ────────── requires T2.2, T3.1 ─────┤
  T3.5  Non-Interactive Protocol ──────── requires T3.3, T3.4 ────┤
  T3.6  Phase 3 Accuracy Gate ────────── requires T3.4, T3.5 ─────┤
                                                                    │
Phase 4 (Months 6-12)                                               │
  T4.1  PrivCirNet Circulant Weights ── requires T3.1 ────────────┤
  T4.2  HE + MPC Hybrid ───────────── requires T3.1 ──────────────┤
  T4.3  TEE + HE Hybrid ───────────── independent ────────────────┤
  T4.4  HE-Optimized Student Model ── requires T3.4 ──────────────┘
```

**Critical path:** T1.1 -> T2.2 -> T2.3 -> T3.1 -> T3.2/T3.3 -> T3.5

---

## 5. Phase 1: Foundation & Quick Wins (Weeks 1-2)

**Target:** Establish measurement infrastructure, fix operational gaps, ~1.5-2x prefill speedup

### T1.1 -- Benchmark Infrastructure

| | |
|---|---|
| **Why** | No benchmarks exist. Every subsequent optimization is unmeasurable without this. **This is the #1 Phase 1 deliverable.** |
| **Effort** | 3-4 days |
| **Risk** | Low |
| **Assignee** | Engineer B |

**Create:**
| File | Purpose |
|---|---|
| `benchmarks/bench_e2e.py` | Seconds-per-token for 1, 5, 11, 22 encrypted layers |
| `benchmarks/bench_he_matmul.py` | Time `he_matmul()` at each real dimension (2048x2048, 2048x256, 2048x5632-split, 5632x2048-split) |
| `benchmarks/bench_serialization.py` | Encrypt/serialize/compress/decompress/deserialize/decrypt cycle per ciphertext |
| `benchmarks/bench_network.py` | Per-operation round-trip latency and payload sizes |
| `benchmarks/conftest.py` | Shared CKKS context and model weight fixtures |
| `benchmarks/results/` | JSON output directory |
| `benchmarks/run_all.sh` | Runner script |

**Output format:**
```json
{"timestamp": "...", "git_sha": "...", "ckks_params": {...}, "results": [{"name": "...", "mean_ms": 0, "std_ms": 0, "samples": 10}]}
```

**Leverage existing instrumentation:**
- `layer_protocol.py:110-187` already measures serialization/network/deserialization per round
- `client.py:104-112` stats dict already tracks per-phase timing
- Extend these to write structured JSON to `benchmarks/results/`
- Reuse `common/logging_utils.py:timed_execution()` context manager

**Acceptance:** `python benchmarks/bench_he_matmul.py` produces per-dimension timings. Results reproducible within 10% variance.

---

### T1.2 -- Session Cleanup

| | |
|---|---|
| **Why** | `session_handler.py:12` stores sessions in `_sessions: dict` with no eviction. Each TenSEAL Context with Galois keys is ~50-100MB. Server will OOM under sustained use. |
| **Effort** | 1 day |
| **Risk** | Low |
| **Assignee** | Engineer B |

**Changes:**

| File | Change |
|---|---|
| `server/handlers/session_handler.py` | Add `SessionEntry` dataclass `(context, created_at, last_accessed)`. Add `cleanup_expired_sessions(max_age_seconds)`. Add `DELETE /api/session/{session_id}` endpoint. Update `last_accessed` in `get_session()` on every call. |
| `server/server.py:28-36` | Register periodic cleanup in `lifespan()` via `asyncio.create_task` (every 60s) |
| `server/config/server_config.yaml` | Add `max_sessions: 50`, `session_ttl_seconds: 3600` |

**Key detail:** Updating `last_accessed` on every `get_session()` call prevents active sessions from expiring mid-generation.

**Acceptance:** 100 rapid session creates + 60s wait -> memory returns to baseline. Access after TTL -> 404.

---

### T1.3 -- Multi-Token Packing

**Status update (2026-03-29): Deferred.**
The original Phase 1 assumption was incorrect for the current TenSEAL backend. Concatenation-style token packing changes the encrypted vector's logical length, which breaks the current `CKKSVector.mm()` weight contract. See [T1.3 spike findings](T1.3-MULTI-TOKEN-PACKING-SPIKE.md). Runtime inference remains un-packed.

| | |
|---|---|
| **Why** | Hidden state (2048) occupies 50% of CKKS slots (4096). During prefill (seq_len > 1), packing 2 tokens per ciphertext (2x2048=4096 = 100% utilization) halves ciphertext count and HE operations. |
| **Effort** | 3-4 days |
| **Risk** | Medium -- off-by-one in pack/unpack can silently corrupt results |
| **Assignee** | Engineer A |

**Changes:**

| File | Change |
|---|---|
| `client/inference/layer_protocol.py` | Add `_pack_tokens(vectors: list[np.ndarray]) -> list[tuple[ts.CKKSVector, int]]` and `_unpack_tokens(dec_vec, pack_count, dim)`. Modify `process_layer():189-305` encryption/decryption loops in all 4 rounds. |
| `server/handlers/inference_handler.py` | Pass `pack_counts` metadata through binary endpoint (informational only) |

**Revised finding:** This is **not valid** for the current TenSEAL `CKKSVector.mm()` path. The server is not oblivious to concatenation-style packing because matmul depends on the encrypted vector's logical size.

**Current disposition:** Defer true multi-token packing to a later phase with either server-assisted unpacking, a SIMD-aware packed matmul design, or a lower-level HE backend.

**Acceptance:** Completed only when a backend-correct packed matmul path exists and shows the target prefill speedup without accuracy regressions.

---

### T1.4 -- Selective Galois Keys (Investigation Only)

**Status update (2026-03-29): Investigated, defer to T3.1 / OpenFHE.**
The current TenSEAL Python API used in this repo documents `context.generate_galois_keys()` but does not provide a documented selective-steps path that we can safely adopt in Phase 1. See [T1.4 finding](T1.4-SELECTIVE-GALOIS-KEYS-FINDING.md).

| | |
|---|---|
| **Why** | Spike task. Research doc claims 50-70% key reduction -- actually ~15% at N=8192. Investigate TenSEAL API for `generate_galois_keys(steps=[...])`. If unsupported, defer to T3.1 (OpenFHE has full support). |
| **Effort** | 0.5 days |
| **Risk** | Low |
| **Assignee** | Engineer A |

**File:** `client/encryption/ckks_context.py:17`

**Result:** No safe/documented TenSEAL Phase 1 implementation found. Defer to T3.1 / OpenFHE or lower-level SEAL access.

---

### T1.5 -- CI Skeleton + CLAUDE.md

| | |
|---|---|
| **Why** | No CI/CD exists. Team of 2-3 needs automated test gates and onboarding documentation. |
| **Effort** | 1 day |
| **Risk** | Low |
| **Assignee** | Engineer B |

**Create:**
| File | Purpose |
|---|---|
| `CLAUDE.md` | Project conventions, architecture, test instructions, CKKS parameter rationale |
| `.github/workflows/test.yml` | `pytest -m "not slow"` on push/PR |
| `.github/workflows/benchmark.yml` | Manual trigger, runs benchmarks, posts summary as PR comment |

---

### T1.6 -- Scale Factor Reduction Experiment

**Status update (2026-03-29): Executed.**
Initial empirical sweep data shows no meaningful speedup from lowering the scale, while approximation error increases materially. Keep `2^40` as the default. See [T1.6 report](T1.6-SCALE-FACTOR-EXPERIMENT-REPORT.md).

| | |
|---|---|
| **Why** | Research doc suggests 2^40 -> 2^30 for 1.2-1.5x speedup. Our corrected estimate: 1.05-1.15x. This is an **experiment**, not a commitment. |
| **Effort** | 1-2 days |
| **Risk** | Low |
| **Assignee** | Engineer A |

**Approach:**
1. Run `bench_he_matmul.py` at scales 2^40, 2^35, 2^30
2. Run `test_e2e_accuracy.py` at each scale
3. Measure matmul error growth at 2048x2048

**Result:** No-go for changing the default Phase 1 scale. Keep `2^40`.

---

### Phase 1 Team Allocation

| Week | Engineer A (HE Specialist) | Engineer B (Systems/Infra) |
|---|---|---|
| 1 | T1.3 Multi-Token Packing (3-4d) | T1.1 Benchmark Infrastructure (3-4d) |
| 2 | T1.4 Galois Keys (0.5d) + T1.6 Scale Experiment (1-2d) | T1.2 Session Cleanup (1d) + T1.5 CI/CLAUDE.md (1d) |

Both tracks run in parallel. Phase 1 completes in ~2 weeks.

---

## 6. Phase 2: Algorithmic Improvements (Weeks 3-6)

**Target:** 10-30x cumulative speedup, reduce round-trips from 4 to 3 per layer

### T2.1 -- Intel HEXL Enablement

| | |
|---|---|
| **Why** | AVX512 detection exists (`server.py:15-25`) but HEXL is not linked. Recompiling SEAL with `-DSEAL_USE_INTEL_HEXL=ON` provides 2-4x acceleration on NTT/modular arithmetic with zero application code changes. |
| **Effort** | 2-3 days |
| **Risk** | Medium (build system complexity) |
| **Depends on** | T1.1 |
| **Assignee** | Engineer B |

**Approach:**
1. Create `scripts/build_tenseal_hexl.sh` -- clone TenSEAL + SEAL, cmake with HEXL, build wheel
2. Enhance `_check_hexl()` in `server.py` to verify HEXL is actually linked (not just AVX512 detected)
3. Document in `docs/intel-hexl-build.md`

**Constraint:** Requires Intel Ice Lake or later Xeon (AVX512-IFMA). Maintain fallback to standard pip TenSEAL.

**Acceptance:** `bench_he_matmul.py` shows measurable improvement on AVX512 hardware. All tests pass.

---

### T2.2 -- Polynomial SiLU Approximation + Round 3/4 Merge

**Status update (2026-03-29): Phase A complete.**
Client-side `poly_silu()` is now implemented and enabled as the default FFN activation approximation, with max approximation error below `0.1` on `[-5, 5]`. The merged-FFN server/client protocol path is implemented behind `use_merged_ffn`, but remains disabled until `T2.3` lands the deeper CKKS chain required to execute it safely.

| | |
|---|---|
| **Why** | SiLU (`nonlinear_ops.py:21`) forces a client round-trip. Polynomial approximation enables server-side computation, merging rounds 3+4 (4->3 round-trips = 25% reduction per layer). |
| **Effort** | 5-7 days |
| **Risk** | High (accuracy, CKKS depth budget) |
| **Depends on** | T1.1, T1.6 |
| **Assignee** | Engineer A |

**CKKS Depth Constraint (Critical):**
```
Gate matmul:           1 depth
poly_silu(Gate):      +1 depth  (degree-2: 0.5x + 0.25x^2)
silu(gate) * up:      +1 depth
Down matmul:          +1 depth
─────────────────────────────
Total:                 4 depths  >>  Current budget: 2
```

This **exceeds** N=8192 capacity. Must be phased:

**Phase A (immediate):** Implement `poly_silu()` as client-side replacement in `nonlinear_ops.py`. Validate accuracy. No round-trip savings yet.

**Phase B (after T2.3):** With N=16384 and depth-5 chain, enable server-side merged FFN.

**Changes:**

| File | Change |
|---|---|
| `server/inference/he_ops.py` | Add `poly_silu(enc_vec) -> ts.CKKSVector`, add `compute_ffn_merged()` |
| `client/inference/nonlinear_ops.py` | Add `poly_silu()` reference implementation for accuracy comparison |
| `client/inference/layer_protocol.py:260-304` | Merge rounds 3+4 behind `use_merged_ffn` flag |
| `server/handlers/inference_handler.py` | Add `"ffn_merged"` operation handler |

**Acceptance:** (1) `poly_silu()` max error < 0.1 on [-5, 5] vs exact SiLU. (2) 3-round protocol matches 4-round output on 3 test prompts.

---

### T2.3 -- Deeper CKKS Parameters (N=16384)

**Status update (2026-03-29): Implemented.**
The repo now uses `poly_modulus_degree=16384` with a depth-5 coefficient chain and a shared `common/constants.py` slot-count constant. FFN intermediate width (`5632`) now fits in a single ciphertext, and the merged FFN path is enabled by default. The merged HE path currently uses a quadratic SiLU approximation to stay within the available scale budget.

| | |
|---|---|
| **Why** | Doubles slots to 8192, enables depth-5 chain for merged FFN (T2.2). FFN intermediate (5632) now fits in 1 ciphertext (5632 < 8192), **eliminating all split-output/split-input logic**. |
| **Effort** | 3-4 days |
| **Risk** | High (4x slower per-op, 3.5x larger ciphertexts) |
| **Depends on** | T2.2 |
| **Assignee** | Engineer A |

**Proposed chain:** `[60, 40, 40, 40, 40, 40, 60]` = 320 bits, depth 5 (under 438-bit security budget at 128-bit security).

**Changes:**

| File | Change |
|---|---|
| Both config YAMLs | Update poly_modulus_degree, coeff_mod_bit_sizes |
| `common/constants.py` (new) | `SLOT_COUNT = poly_modulus_degree // 2` -- **refactor out of 3 hardcoded locations** (`he_ops.py:16`, `layer_protocol.py:41`, `weight_manager.py:65`) |
| `server/model/weight_manager.py:80-91` | Split logic for FFN dims becomes unnecessary (5632 < 8192) |

**Critical tradeoff:** Each HE op at N=16384 is ~4x slower than N=8192. But:
- FFN no longer needs splitting (saves 2 extra matmul calls per token)
- Round merging (T2.2) reduces round-trips from 4 to 3
- 4x token packing during prefill (4*2048 = 8192 slots)
- **Net effect must be benchmarked before committing**

**Acceptance:** All accuracy tests pass at N=16384. `SLOT_COUNT` no longer hardcoded. Net benchmark improvement for prefill.

---

### T2.4 -- WebSocket Persistent Connection

**Status update (2026-03-29): Implemented.**
The client protocol now supports a persistent WebSocket transport using the same msgpack+zstd payloads as the HTTP binary endpoint, with HTTP fallback kept for compatibility.

| | |
|---|---|
| **Why** | Eliminates per-round HTTP overhead, enables server push for pipelining (T2.5). |
| **Effort** | 3-4 days |
| **Risk** | Medium |
| **Depends on** | T1.1 |
| **Assignee** | Engineer B |

**Changes:**

| File | Change |
|---|---|
| `client/inference/layer_protocol.py:100` | Add WebSocket branch in `_send_request()`, option in `__init__` |
| `server/server.py` | Add WebSocket route |
| `server/handlers/inference_handler.py` | Add WebSocket handler reusing existing dispatch logic |

Keep HTTP fallback for backward compatibility.

**Acceptance:** E2E test passes over WebSocket. Measurable latency reduction for 5+ encrypted layers.

---

### T2.5 -- Async Client-Server Pipelining

**Status update (2026-03-29): Implemented.**
The client now has an async-capable encrypted-layer runner and an opt-in pipelined execution path that reuses pre-fetched encrypted-layer metadata and drives encrypted layer processing through `process_layer_async()`. The protocol remains dependency-correct and preserves a synchronous fallback.

**Live update (2026-04-11):**
`process_layer_async()` now executes a dedicated async encrypted-layer flow (instead of delegating to sync), and transport/bench harnesses now allow explicit `http|websocket` and `sync|async` mode selection for controlled measurements. Minimal e2e artifacts were added:

- `benchmarks/results/bench_e2e_sync_min.json`
- `benchmarks/results/bench_e2e_async_min.json`

| | |
|---|---|
| **Why** | Overlap client decrypt/process for layer N-1 with server compute for layer N. Approaches `max(server_time, client_time)` instead of `sum(all)`. |
| **Effort** | 4-5 days |
| **Risk** | Medium (correctness) |
| **Depends on** | T2.4 |
| **Assignee** | Engineer B |

**Changes:**

| File | Change |
|---|---|
| `client/inference/layer_protocol.py:189` | Async-capable `process_layer()` |
| `client/client.py:177-186` | Pipelined layer loop using `asyncio` or `concurrent.futures` |

**Note:** Intra-layer pipelining is impossible (each round depends on the previous). Inter-layer pipelining works: start layer N+1 encryption while receiving layer N results.

**Acceptance:** Identical tokens with and without pipelining. Wall-clock improvement for 5+ encrypted layers.

---

### T2.6 -- Phase 2 Accuracy Gate

**Status update (2026-03-29): Conditional go.**
Focused Phase 2 accuracy and transport checks passed, and the `N=16384` merged FFN path is numerically correct relative to its quadratic HE target. The open quality risk is that the merged HE path uses a lower-accuracy quadratic SiLU approximation than the degree-6 client reference. See [T2.6 gate report](T2.6-PHASE-2-ACCURACY-GATE.md).

**Live update (2026-04-11):**
Prompt-level comparisons and per-operation transport artifacts were added. On short-prompt samples (`1` token, `1` encrypted layer):

- `exact_split` vs `poly_split`: exact matches in sampled runs
- `exact_split` vs `merged_he`: divergences observed in sampled runs

Artifacts:

- `benchmarks/results/phase2_accuracy_gate_live_exact_vs_poly_two_short.json`
- `benchmarks/results/phase2_accuracy_gate_live_exact_vs_merged_two_short.json`
- `benchmarks/results/phase2_accuracy_gate_live_exact_vs_poly_five_short.json`
- `benchmarks/results/phase2_accuracy_gate_live_exact_vs_merged_five_short.json`
- `benchmarks/results/bench_network_http_qkv.json`
- `benchmarks/results/bench_network_http_remaining_ops.json`
- `benchmarks/results/bench_network_websocket_qkv.json`
- `benchmarks/results/bench_network_websocket_remaining_ops.json`

**Live update (2026-04-12):**
An explicit gate-decision artifact was added:

- `benchmarks/results/phase2_gate_decision_report.json`

The report records `decision=conditional_go` with `systems_go=true` and `quality_parity_go=false` based on current saved artifacts.

| | |
|---|---|
| **Why** | Validate all Phase 2 changes maintain acceptable accuracy before Phase 3's invasive migration. |
| **Effort** | 2 days |
| **Risk** | Low |
| **Depends on** | T2.2, T2.3 |
| **Assignee** | Engineer C |

**Checks:**
1. Encrypted-vs-plaintext accuracy at N=16384 on layers 0-4, 5 test prompts
2. Perplexity delta: poly_silu vs exact SiLU on 100-sentence eval set
3. Maximum observed error per layer documented

**Acceptance:** Written go/no-go report for Phase 3.

---

### Phase 2 Team Allocation

| Weeks | Engineer A (HE) | Engineer B (Systems) | Engineer C |
|---|---|---|---|
| 3-4 | T2.2 Poly SiLU (5-7d) | T2.1 HEXL Enablement (2-3d) | -- |
| 5 | T2.3 Deeper CKKS (3-4d) | T2.4 WebSocket (3-4d) | -- |
| 6 | -- | T2.5 Async Pipeline (4-5d) | T2.6 Accuracy Gate (2d) |

---

## 7. Phase 3: Architecture Migration (Months 2-5)

**Target:** 100-500x cumulative speedup, sub-second per-token at 22 encrypted layers

### T3.1 -- TenSEAL -> OpenFHE Migration

**Status update (2026-04-12): OpenFHE vector path and initial matmul landed; still experimental.**
Core runtime modules already route through [`common/he_backend.py`](common/he_backend.py), and the backend now has an explicit OpenFHE path for context creation, public-context serialization, encrypt/decrypt, vector serialization, cloning, `square()`, and an initial `matmul()` implementation (`EvalInnerProduct` + `EvalMerge`). A local smoke probe artifact now passes for these operations (`benchmarks/results/openfhe_probe.json`), but production inference remains TenSEAL-default until larger-dimension benchmark and parity validation is complete. See [T3.1 environment finding](T3.1-OPENFHE-ENV-FINDING.md) and the [OpenFHE matmul implementation plan](T3.1-OPENFHE-MATMUL-PLAN.md).

Initial backend-level smoke benchmark artifact:

- `benchmarks/results/bench_he_matmul_backend_openfhe_smoke.json`
- `benchmarks/results/compare_he_backend_matmul_smoke.json`
- `benchmarks/results/compare_he_backend_matmul_medium.json`
- `benchmarks/results/compare_he_backend_matmul_cpu_repro.json`
- `benchmarks/results/compare_he_backend_matmul_512x256.json`
- `benchmarks/results/compare_he_backend_matmul_1024x256.json`
- `benchmarks/results/compare_he_backend_matmul_2048x256.json`
- `benchmarks/results/compare_he_backend_matmul_2048x256_samples2.json`
- `benchmarks/results/compare_he_backend_matmul_2048x256_samples2_keycache.json`
- `benchmarks/results/compare_he_backend_matmul_2048x256_warm.json`
- `benchmarks/results/compare_he_backend_matmul_tenseal_2048x1024.json`
- `benchmarks/results/compare_he_backend_matmul_openfhe_2048x1024.json`
- `benchmarks/results/compare_he_backend_matmul_tenseal_2048x2048.json`
- `benchmarks/results/compare_he_backend_matmul_openfhe_2048x2048.json`

Current signal from these artifacts: numerical parity is strong on sampled dimensions, but OpenFHE runtime is still far slower than TenSEAL in the current matmul implementation (for example, ~32x slower at `2048x2048`; `2048x256` remains in the ~8x range across cache-optimized samples).

Readiness artifact:

- `benchmarks/results/t3_openfhe_matmul_readiness.json` (`decision=no_go` for production default under current thresholds)

| | |
|---|---|
| **Why** | TenSEAL is CPU-only, no bootstrapping, no diagonal packing, limited maintenance since 2022. OpenFHE supports all of these plus GPU extensions. This is the **most critical Phase 3 prerequisite**. |
| **Effort** | 3-4 weeks |
| **Risk** | High (OpenFHE Python bindings less mature, manual matmul may initially be slower) |
| **Depends on** | T2.1, T2.3 |
| **Assignee** | Engineer A |

**Approach:**
1. **Create abstraction layer:** `common/he_backend.py` with functions: `create_context()`, `encrypt()`, `decrypt()`, `serialize()`, `deserialize()`, `matmul()`. Implement with TenSEAL first.
2. **Install OpenFHE:** `openfhe` Python bindings (pip or source build with GPU support)
3. **Port incrementally:** One function at a time, running test suite after each
4. **Biggest change:** `he_matmul()` -- TenSEAL's `.mm()` is a single call; OpenFHE requires manual `EvalMult + EvalRotate + EvalAdd` loops

**Files (major rewrite):** `ckks_context.py`, `layer_protocol.py`, `he_ops.py`, `inference_handler.py`, `session_handler.py`, all test files referencing `tenseal`

**Mitigation:** Keep TenSEAL codepath as fallback via `he_backend.py` abstraction. Run accuracy tests after every sub-module port.

**Acceptance:** All tests pass on OpenFHE backend. E2E within 20% of TenSEAL baseline. Bootstrapping API accessible.

---

### T3.2 -- Diagonal Packing MatMul (THOR-style)

**Status update (2026-04-13): Preprocessing and runtime prototype landed; optimized runtime still blocked.**
Per-layer cyclic diagonal extraction and caching are implemented in `weight_manager.py`, and a correctness-oriented `he_matmul_diagonal()` runtime prototype is now present in `he_ops.py`. The optimized packed-matmul runtime remains blocked on larger-dimension OpenFHE parity/performance validation plus packed-kernel implementation work. See [T3.2 status](T3.2-DIAGONAL-PACKING-STATUS.md).

Prototype benchmark artifact:

- `benchmarks/results/bench_diagonal_runtime_64.json`

| | |
|---|---|
| **Why** | Standard CKKS matrix-vector multiply: O(D) rotations. Baby-step/giant-step diagonal packing: O(sqrt(D)). For D=2048: ~45 rotations vs ~2048. **Corrected estimate: 5-15x** matmul speedup. |
| **Effort** | 2-3 weeks |
| **Risk** | High (complex implementation, precision differences) |
| **Depends on** | T3.1 |
| **Assignee** | Engineer A |

**Design:**
1. At model load (`weight_manager.py`): precompute D_in diagonal vectors per weight matrix
2. Encode each diagonal as a CKKS plaintext (one-time cost)
3. At inference: for each diagonal, rotate input ciphertext, multiply by plaintext, accumulate
4. Baby-step/giant-step: choose b = ceil(sqrt(D)). Total rotations: 2*sqrt(D)

**Acceptance:** `he_matmul` at 2048x2048 is at least 5x faster than TenSEAL baseline. Accuracy within 2x of naive matmul error.

---

### T3.3 -- GPU-Accelerated CKKS

**Status update (2026-04-14): Hardware probe available; feasibility still blocked.**
Backend status reporting now captures OpenFHE and GPU probe results, and T3.3 artifacts are generated at `benchmarks/results/t3_gpu_readiness.json` and `benchmarks/results/t3_gpu_feasibility.json`. Current decision remains `no_go`: T3.1 readiness is `no_go`, and feasibility probe evidence does not yet show a usable GPU HE path in the current OpenFHE runtime. See [T3.3 GPU status](T3.3-GPU-STATUS.md).

| | |
|---|---|
| **Why** | GPU provides 20-50x speedup for leveled CKKS ops (corrected from 100-200x). With bootstrapping, reaches 100-200x. |
| **Effort** | 3-4 weeks |
| **Risk** | High (CUDA build chain, GPU memory management) |
| **Depends on** | T3.1 |
| **Assignee** | Engineer B |

**Options:** OpenFHE GPU extension, Phantom (`github.com/encryptorion-lab/phantom-fhe`), or FIDESlib (builds on Phantom).

**Requires:** CUDA-capable GPU on server. Add GPU backend option to `common/he_backend.py`.

**Acceptance:** `bench_he_matmul.py` shows >20x improvement over CPU OpenFHE. All accuracy tests pass.

---

### T3.4 -- Full Polynomial Model

**Status update (2026-04-13): Still blocked; readiness artifact added.**
The repo has Phase 2 SiLU approximations plus prototype client-side `poly_rms_norm` / `poly_softmax`, but server-side RMSNorm and softmax polynomial replacements are still missing. The current polynomial readiness artifact remains `no_go` (`benchmarks/results/t3_polynomial_readiness.json`). See [T3.4 status](T3.4-FULL-POLYNOMIAL-MODEL-STATUS.md).

| | |
|---|---|
| **Why** | Approximate ALL non-linear ops server-side, reducing protocol to 1-2 rounds/layer. |
| **Effort** | 4-6 weeks |
| **Risk** | Very high (polynomial softmax is notoriously difficult) |
| **Depends on** | T2.2, T3.1 |
| **Assignee** | Engineer A |

**Operations to approximate:**
| Operation | Current Location | Polynomial Strategy |
|---|---|---|
| RMSNorm | `nonlinear_ops.py:10-18` | Inverse square root polynomial (Goldschmidt iteration) |
| Softmax | `nonlinear_ops.py:26-30` | Polynomial or PowerFormer (replace exp with power function) |
| SiLU | `nonlinear_ops.py:21-23` | Already done in T2.2 |
| RoPE | `nonlinear_ops.py:83-164` | Precomputable as plaintext rotations (no approximation needed) |

**Acceptance:** Perplexity increase < 5% vs exact model on 1000-sentence eval set.

---

### T3.5 -- Non-Interactive Protocol

**Status update (2026-04-13): Blocked by readiness gates; artifact added.**
The one-request / one-response protocol is not implementable yet because both GPU-backed HE execution and the full polynomial model remain `no_go` in current readiness artifacts. See [T3.5 status](T3.5-NON-INTERACTIVE-PROTOCOL-STATUS.md) and `benchmarks/results/t3_noninteractive_readiness.json`.

| | |
|---|---|
| **Why** | Client sends `Enc(embeddings)`, server processes all 22 layers homomorphically, returns `Enc(logits)`. Zero round-trips. |
| **Effort** | 2-3 weeks |
| **Risk** | Very high (depends on T3.3 + T3.4 both succeeding) |
| **Depends on** | T3.3, T3.4 |
| **Assignee** | Engineer C |

**Acceptance:** Client sends one request, receives one response, produces correct tokens.

---

### T3.6 -- Phase 3 Accuracy Gate

**Status update (2026-04-13): No-go for Phase 4 (artifact-backed).**
Phase 3 still does not meet backend, polynomial-model, and non-interactive readiness gates; see `benchmarks/results/t3_phase3_gate.json` and [T3.6 gate report](T3.6-PHASE-3-ACCURACY-GATE.md).

Decision thresholds are documented in [T3-DECISION-CRITERIA.md](T3-DECISION-CRITERIA.md).

| | |
|---|---|
| **Effort** | 3 days |
| **Depends on** | T3.4, T3.5 |
| **Assignee** | Engineer C |

**Checks:**
1. Perplexity within 10% of plaintext baseline
2. Token-level agreement > 80% with plaintext model on 100 test prompts
3. Per-layer error documented for all 22 layers

**Acceptance:** Written go/no-go report for Phase 4.

---

## 8. Phase 4: Research Frontier (Months 6-12)

| Task | Description | Depends On | Effort | Risk |
|---|---|---|---|---|
| **T4.1 PrivCirNet** | Block circulant weight matrices. O(1) rotation matmul instead of O(sqrt(D)). Requires model retraining with circulant constraints. | T3.1 | 6-8 weeks | Very High |
| **T4.2 HE+MPC Hybrid** | Garbled circuits or secret sharing for non-linear ops (BOLT/CipherFormer-style). Better accuracy than polynomial approx, 7-11x communication reduction. | T3.1 | 8-12 weeks | Very High |
| **T4.3 TEE+HE Hybrid** | Intel SGX/TDX or AMD SEV for non-linear ops in secure enclave. No accuracy loss, 2-4x improvement. Weaker security guarantees than pure HE. | Independent | 6-8 weeks | High |
| **T4.4 Student Model** | Distill TinyLlama into 4-8 layer, 512-1024 dim model with polynomial activations. Purpose-built for HE inference. 10-50x from reduced dimensions. | T3.4 | 8-12 weeks | High |

---

## 9. Performance Projections

| Milestone | Encrypted Layers | Est. Time/Token | Improvement vs Current |
|---|---|---|---|
| **Current baseline** | 1/22 | 3-8s | 1x |
| After Phase 1 | 1/22 | 2-5s (prefill ~1.5x faster) | ~1.5x |
| After Phase 2 | 1/22 | 0.3-1s | ~5-15x |
| After Phase 2 | 22/22 | 5-20s | First time feasible |
| After Phase 3 | 22/22 | 0.1-1s | ~100-300x |
| After Phase 3+4 | 22/22 | <0.5s | ~300-500x |

---

## 10. Verification Strategy

### After Every Task

1. `pytest -m "not slow"` -- all unit tests green
2. `python benchmarks/bench_he_matmul.py` -- no performance regression on unrelated paths
3. `python benchmarks/bench_e2e.py --layers 1` -- E2E latency measured
4. Results committed to `benchmarks/results/{date}_{task_id}.json`

### At Each Phase Gate (T2.6, T3.6)

1. Full accuracy suite including `test_e2e_accuracy.py`
2. Perplexity comparison on 100-sentence eval set (encrypted vs plaintext)
3. Written go/no-go report with quantified error metrics before proceeding

### Regression Tracking

- All benchmark results stored as JSON in `benchmarks/results/`
- Each result tagged with git SHA and CKKS parameters
- Phase-over-phase comparison automated in CI (`.github/workflows/benchmark.yml`)

---

## 11. Critical Files Reference

| File | Lines | Touched By Tasks |
|---|---|---|
| `client/inference/layer_protocol.py` | 305 | T1.3, T2.2, T2.4, T2.5, T3.1 |
| `server/inference/he_ops.py` | 192 | T2.1, T2.2, T3.1, T3.2, T3.3 |
| `server/model/weight_manager.py` | 96 | T2.3, T3.2 |
| `server/handlers/inference_handler.py` | 260 | T2.2, T2.4, T3.1 |
| `server/handlers/session_handler.py` | 43 | T1.2, T3.1 |
| `client/encryption/ckks_context.py` | 37 | T1.4, T2.3, T3.1 |
| `client/inference/nonlinear_ops.py` | 210 | T2.2, T3.4 |
| `client/client.py` | 290 | T2.5, T3.5 |
| `common/logging_utils.py` | 38 | T1.1 (reuse `timed_execution()`) |
| `server/server.py` | 45 | T1.2, T2.1, T2.4, T3.3 |

---

*Plan prepared 2026-03-29. Update this document as optimizations are implemented and benchmarked.*
