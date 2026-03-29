"""Benchmark ciphertext serialization and transport-adjacent operations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tenseal as ts
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.common import benchmark_operation, seeded_rng, summarize_samples, write_benchmark_report
from client.encryption.ckks_context import create_ckks_context

_zstd_compressor = zstd.ZstdCompressor(level=3)
_zstd_decompressor = zstd.ZstdDecompressor()


def _encrypt(context: ts.Context, vector: np.ndarray) -> ts.CKKSVector:
    return ts.ckks_vector(context, vector.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CKKS serialization pipeline.")
    parser.add_argument("--samples", type=int, default=10, help="Number of measured samples.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_serialization.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    context = create_ckks_context()
    rng = seeded_rng(args.seed)
    vector = rng.normal(0.0, 0.01, size=2048).astype(np.float32)
    enc_vector = _encrypt(context, vector)
    serialized = enc_vector.serialize()
    compressed = _zstd_compressor.compress(serialized)

    results = [
        summarize_samples(
            "encrypt_2048",
            benchmark_operation(lambda: _encrypt(context, vector), args.samples, args.warmups),
        ),
        summarize_samples(
            "serialize_compress_2048",
            benchmark_operation(
                lambda: _zstd_compressor.compress(enc_vector.serialize()),
                args.samples,
                args.warmups,
            ),
            metadata={"serialized_bytes": len(serialized), "compressed_bytes": len(compressed)},
        ),
        summarize_samples(
            "decompress_deserialize_2048",
            benchmark_operation(
                lambda: ts.ckks_vector_from(context, _zstd_decompressor.decompress(compressed)),
                args.samples,
                args.warmups,
            ),
        ),
        summarize_samples(
            "decrypt_2048",
            benchmark_operation(lambda: enc_vector.decrypt(), args.samples, args.warmups),
        ),
        summarize_samples(
            "full_cycle_2048",
            benchmark_operation(
                lambda: ts.ckks_vector_from(
                    context,
                    _zstd_decompressor.decompress(
                        _zstd_compressor.compress(_encrypt(context, vector).serialize())
                    ),
                ).decrypt(),
                args.samples,
                args.warmups,
            ),
        ),
    ]

    output_path = write_benchmark_report(
        args.output,
        results,
        metadata={"vector_dim": 2048, "seed": args.seed},
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
