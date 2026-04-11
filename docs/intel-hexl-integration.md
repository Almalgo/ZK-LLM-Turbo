# Intel HEXL Integration Guide

This repository supports Intel HEXL as a build-time TenSEAL optimization.

- There is no application-level `--hexl` runtime flag.
- HEXL detection is automatic at server startup.
- A standalone linkage probe script is provided for CI and local verification.

## Recommended Workflow

1. Build a HEXL-enabled TenSEAL wheel with the repo script:

```bash
scripts/build_tenseal_hexl.sh
```

2. Install the built wheel into your target environment:

```bash
python -m pip install --force-reinstall dist/hexl/tenseal-*.whl
```

3. Verify linkage using the probe script:

```bash
python scripts/probe_hexl_linkage.py --json
```

Optionally enforce linkage in automation:

```bash
python scripts/probe_hexl_linkage.py --require-linked
```

4. Run runtime check and benchmark:

```bash
python -m server.server
python -m benchmarks.bench_he_matmul
```

Compare matmul numbers against the non-HEXL baseline on the same host.

## Notes

- CKKS defaults in this repo are currently `N=16384` and `[60, 40, 40, 40, 40, 40, 60]`.
- HEXL speedups are hardware-dependent and should always be validated with benchmark artifacts.
- For detailed build steps and local-prefix layout, see `docs/intel-hexl-build.md`.
