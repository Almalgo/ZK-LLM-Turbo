# Intel HEXL Build Guide

## Goal

Build a HEXL-enabled TenSEAL wheel for environments where:

- the CPU supports AVX512
- the team wants to benchmark whether Intel HEXL improves CKKS matmul latency
- the standard `pip install tenseal` wheel is not sufficient

Additional local build prerequisites:

- Python development headers for your interpreter (for example `python3.12-dev`)
- A CMake version compatible with Intel HEXL/SEAL dependency tree (3.x recommended)

This repository keeps the default pip-installed TenSEAL path as the fallback.

## Quick Check

Check CPU support:

```bash
grep -c avx512 /proc/cpuinfo
```

If this prints `0`, do not expect Intel HEXL to help on this machine.

Check the current runtime probe:

```bash
python -m server.server
python scripts/probe_hexl_linkage.py --json
```

At startup the server now reports one of:

- HEXL linked
- AVX512 present but current TenSEAL build does not appear HEXL-linked
- AVX512 not detected

## Build Script

Use the repo script:

```bash
scripts/build_tenseal_hexl.sh
```

What it does:

1. clones Intel HEXL
2. clones Microsoft SEAL
3. clones TenSEAL
4. builds HEXL locally
5. builds SEAL with `-DSEAL_USE_INTEL_HEXL=ON`
6. builds a TenSEAL wheel against that local toolchain

Defaults:

- `HEXL_REF=v1.2.5`
- `SEAL_REF=v4.1.1`
- `TENSEAL_REF=v0.3.15`

Outputs:

- local install prefix under `.local/tenseal-hexl/`
- built wheel under `dist/hexl/`

The script now runs a preflight check for Python development headers (`Python.h`) and fails early with a clear message if they are missing.

## Install The Built Wheel

Inside the target Python environment:

```bash
python -m pip install --force-reinstall dist/hexl/tenseal-*.whl
```

## Validate

1. Start the server and confirm the runtime probe reports `HEXL linked`.
2. Run:

```bash
python -m benchmarks.bench_he_matmul
```

3. Compare results against the non-HEXL baseline from the same machine.

Optional: generate an acceptance report from baseline/candidate benchmark JSONs:

```bash
python benchmarks/report_hexl_acceptance.py \
  --baseline benchmarks/results/bench_he_matmul_baseline.json \
  --candidate benchmarks/results/bench_he_matmul_hexl.json \
  --output benchmarks/results/hexl_acceptance_report.json
```

This combines benchmark speedups with linkage probe status in one artifact.

## Notes

- This is a build-time optimization. The application protocol does not change.
- Keep the standard pip TenSEAL path available as fallback.
- Acceptance for `T2.1` is benchmark improvement on compatible AVX512 hardware, not just a successful build.
