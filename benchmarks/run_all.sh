#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" benchmarks/bench_he_matmul.py
"$PYTHON_BIN" benchmarks/bench_serialization.py
"$PYTHON_BIN" benchmarks/bench_network.py
"$PYTHON_BIN" benchmarks/bench_e2e.py
