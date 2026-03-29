# CLAUDE.md

## Project Overview

ZK-LLM-Turbo is a privacy-preserving split-inference prototype built around:

- TinyLlama 1.1B
- CKKS homomorphic encryption via TenSEAL
- A client/server split where encrypted linear layers run on the server and nonlinear operations run on the client

The current codebase is organized as:

- `client/`: tokenizer, embeddings, CKKS context creation, client-side inference protocol
- `server/`: FastAPI app, session handling, encrypted linear algebra, model weight management
- `common/`: shared logging utilities
- `benchmarks/`: benchmark entrypoints and shared reporting helpers
- `docs/`: plans, findings, and research notes

## Architecture Notes

The active split-inference protocol is implemented in `client/inference/layer_protocol.py` and `server/handlers/inference_handler.py`.

Each encrypted decoder layer currently uses 4 network rounds:

1. Q/K/V projections
2. O projection
3. FFN gate/up projections
4. FFN down projection

Current Phase 1 findings that matter for future work:

- `T1.3` multi-token packing is deferred because concatenation-style packing is incompatible with the current TenSEAL `CKKSVector.mm()` path.
- `T1.4` selective Galois keys is deferred because no safe documented selective-steps API was found in the TenSEAL Python surface used by this repo.

See:

- `docs/ZK-LLM-OPTIMIZATION-ACTION-PLAN.md`
- `docs/T1.3-MULTI-TOKEN-PACKING-SPIKE.md`
- `docs/T1.4-SELECTIVE-GALOIS-KEYS-FINDING.md`

## CKKS Parameters

Default client config lives in `client/config/client_config.yaml`:

- `poly_modulus_degree: 8192`
- `coeff_mod_bit_sizes: [60, 40, 40, 60]`
- `global_scale: 2^40`
- `use_galois_keys: true`
- `use_relin_keys: true`

Why these values:

- `8192` gives `4096` CKKS slots, which is enough for the 2048 hidden dimension with room for split-output handling.
- `[60, 40, 40, 60]` supports the current leveled computation depth without moving to the larger ciphertext sizes of `16384`.
- `2^40` is the current baseline scale for acceptable accuracy under the active Phase 1 design.

Do not casually change these in feature work. Any CKKS parameter change should be treated as an experiment and benchmarked.

## Local Development

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn server.server:app --reload --host 0.0.0.0 --port 8000
```

Run the client:

```bash
python -m client.client --prompt "Hello world"
```

## Tests

Fast test gate used by CI:

```bash
pytest -m "not slow"
```

Other useful commands:

```bash
pytest -v
pytest -m slow
```

Notes:

- `slow` tests may download model assets.
- Benchmarks live outside the standard test gate and should not run on every push.

## Benchmarks

Benchmark entrypoints:

- `benchmarks/bench_he_matmul.py`
- `benchmarks/bench_serialization.py`
- `benchmarks/bench_network.py`
- `benchmarks/bench_e2e.py`

Run all benchmarks:

```bash
benchmarks/run_all.sh
```

Important:

- `bench_he_matmul.py` and `bench_serialization.py` are offline.
- `bench_network.py` and `bench_e2e.py` require a running server.
- Benchmark results are written to `benchmarks/results/`.

## Conventions

- Prefer targeted fixes over broad refactors.
- Keep runtime HE behavior conservative unless benchmarked and validated.
- When a performance idea depends on TenSEAL backend semantics, document the finding before landing speculative protocol changes.
- Treat `docs/` as part of the deliverable for spike tasks and optimization investigations.
