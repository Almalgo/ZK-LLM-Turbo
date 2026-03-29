"""Benchmark HE matrix multiplication at the project's real dimensions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.common import benchmark_operation, load_ckks_params, seeded_rng, summarize_samples, write_benchmark_report
from client.encryption.ckks_context import create_ckks_context
from server.inference.he_ops import he_matmul, he_matmul_split_input, he_matmul_split_output


def _make_vector(context: ts.Context, values: np.ndarray) -> ts.CKKSVector:
    return ts.ckks_vector(context, values.tolist())


def _clone_vector(context: ts.Context, values: np.ndarray) -> ts.CKKSVector:
    """Create a fresh ciphertext for each sample to avoid consuming modulus levels."""
    return _make_vector(context, values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HE matrix multiplication.")
    parser.add_argument("--samples", type=int, default=5, help="Number of measured samples per dimension.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="client/config/client_config.yaml",
        help="Client config path used to create the CKKS context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_he_matmul.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    context = create_ckks_context(config_path=args.config_path)
    rng = seeded_rng(args.seed)
    results = []

    hidden_vector = rng.normal(0.0, 0.01, size=2048).astype(np.float32)
    hidden_weight_square = rng.normal(0.0, 0.01, size=(2048, 2048)).astype(np.float32).tolist()
    hidden_weight_small = rng.normal(0.0, 0.01, size=(2048, 256)).astype(np.float32).tolist()
    hidden_weight_ffn = rng.normal(0.0, 0.01, size=(2048, 5632)).astype(np.float32)
    hidden_weight_ffn_chunks = [
        hidden_weight_ffn[:, :4096].tolist(),
        hidden_weight_ffn[:, 4096:].tolist(),
    ]
    down_input = rng.normal(0.0, 0.01, size=5632).astype(np.float32)
    down_weight = rng.normal(0.0, 0.01, size=(5632, 2048)).astype(np.float32)
    chunk_sizes = [4096, 1536]

    samples = benchmark_operation(
        lambda: he_matmul(
            _clone_vector(context, hidden_vector),
            precomputed_list=hidden_weight_square,
        ),
        samples=args.samples,
        warmups=args.warmups,
    )
    results.append(summarize_samples("he_matmul_2048x2048", samples))

    samples = benchmark_operation(
        lambda: he_matmul(
            _clone_vector(context, hidden_vector),
            precomputed_list=hidden_weight_small,
        ),
        samples=args.samples,
        warmups=args.warmups,
    )
    results.append(summarize_samples("he_matmul_2048x256", samples))

    samples = benchmark_operation(
        lambda: he_matmul_split_output(
            _clone_vector(context, hidden_vector),
            weight_matrix=hidden_weight_ffn,
            precomputed_chunks=hidden_weight_ffn_chunks,
        ),
        samples=args.samples,
        warmups=args.warmups,
    )
    results.append(
        summarize_samples(
            "he_matmul_split_output_2048x5632",
            samples,
            metadata={"chunk_sizes": [4096, 1536]},
        )
    )

    samples = benchmark_operation(
        lambda: he_matmul_split_input(
            [
                _clone_vector(context, down_input[:4096]),
                _clone_vector(context, down_input[4096:]),
            ],
            down_weight,
            chunk_sizes=chunk_sizes,
        ),
        samples=args.samples,
        warmups=args.warmups,
    )
    results.append(
        summarize_samples(
            "he_matmul_split_input_5632x2048",
            samples,
            metadata={"chunk_sizes": chunk_sizes},
        )
    )

    output_path = write_benchmark_report(
        args.output,
        results,
        metadata={"matrix_source": "synthetic", "seed": args.seed},
        ckks_params=load_ckks_params(Path(args.config_path)),
    )

    print(f"Wrote {output_path}")
    for result in results:
        print(f"{result['name']}: mean={result['mean_ms']}ms std={result['std_ms']}ms")


if __name__ == "__main__":
    main()
