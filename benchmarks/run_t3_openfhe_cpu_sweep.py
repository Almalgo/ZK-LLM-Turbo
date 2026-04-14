"""Run a reproducible CPU-only OpenFHE vs TenSEAL benchmark sweep."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible T3 CPU benchmark sweep")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--dims",
        type=str,
        default="512x256,1024x256,2048x256,2048x1024",
        help="Comma-separated D_inxD_out dimensions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/compare_he_backend_matmul_cpu_repro.json"),
    )
    args = parser.parse_args()

    cmd = [
        args.python,
        "benchmarks/compare_he_backend_matmul.py",
        "--backends",
        "tenseal,openfhe",
        "--dims",
        args.dims,
        "--seed",
        str(args.seed),
        "--samples",
        str(args.samples),
        "--warmups",
        str(args.warmups),
        "--output",
        str(args.output),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"sweep failed ({proc.returncode})")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
