"""Run Phase 3 readiness reports in sequence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Phase 3 readiness/gate report scripts")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use")
    args = parser.parse_args()

    py = args.python
    _run([py, "benchmarks/report_t3_openfhe_matmul_readiness.py"])
    _run([py, "benchmarks/report_t3_gpu_readiness.py"])
    _run([py, "benchmarks/report_t3_polynomial_readiness.py"])
    _run([py, "benchmarks/report_t3_noninteractive_readiness.py"])
    _run([py, "benchmarks/report_t3_phase3_gate.py"])

    print("Phase 3 gate artifacts refreshed")


if __name__ == "__main__":
    main()
