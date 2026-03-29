#!/usr/bin/env bash
set -euo pipefail

# Offline benchmarks do not require the FastAPI server.
# Online benchmarks require `python -m server.server` running at the configured base URL.

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" benchmarks/bench_he_matmul.py
"$PYTHON_BIN" benchmarks/bench_serialization.py
"$PYTHON_BIN" benchmarks/bench_network.py
"$PYTHON_BIN" benchmarks/bench_e2e.py
