"""Benchmark diagonal-runtime prototype against dense matmul path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import benchmark_operation, load_ckks_params, seeded_rng, summarize_samples, write_benchmark_report
from client.encryption.ckks_context import create_ckks_context
from server.inference.he_ops import he_matmul, he_matmul_diagonal


def _matrix_to_diagonals(weight_matrix: np.ndarray) -> list[list[float]]:
    rows, cols = weight_matrix.shape
    diagonals = []
    for offset in range(cols):
        diagonals.append([
            float(weight_matrix[row_idx, (row_idx + offset) % cols])
            for row_idx in range(rows)
        ])
    return diagonals


def _make_vector(context: ts.Context, values: np.ndarray) -> ts.CKKSVector:
    return ts.ckks_vector(context, values.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark diagonal-runtime prototype matmul path")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Square matrix dimension for prototype benchmark.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="client/config/client_config.yaml",
        help="Client config path used to create the CKKS context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_diagonal_runtime.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    context = create_ckks_context(config_path=args.config_path)
    rng = seeded_rng(args.seed)
    dim = int(args.dim)

    vector = rng.normal(0.0, 0.01, size=dim).astype(np.float32)
    matrix = rng.normal(0.0, 0.01, size=(dim, dim)).astype(np.float32)
    diagonals = _matrix_to_diagonals(matrix)

    dense_samples = benchmark_operation(
        lambda: he_matmul(_make_vector(context, vector), matrix),
        samples=args.samples,
        warmups=args.warmups,
    )
    diag_samples = benchmark_operation(
        lambda: he_matmul_diagonal(_make_vector(context, vector), diagonals),
        samples=args.samples,
        warmups=args.warmups,
    )

    results = [
        summarize_samples(f"dense_matmul_{dim}x{dim}", dense_samples),
        summarize_samples(f"diagonal_runtime_prototype_{dim}x{dim}", diag_samples),
    ]
    dense_mean = results[0]["mean_ms"]
    diag_mean = results[1]["mean_ms"]
    slowdown = (diag_mean / dense_mean) if dense_mean > 0 else 0.0

    output_path = write_benchmark_report(
        args.output,
        results,
        metadata={
            "seed": args.seed,
            "dim": dim,
            "diagonal_runtime_slowdown_vs_dense": round(float(slowdown), 4),
        },
        ckks_params=load_ckks_params(Path(args.config_path)),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
