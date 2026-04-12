"""Benchmark matmul through backend abstraction (TenSEAL or OpenFHE)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import benchmark_operation, load_ckks_params, seeded_rng, summarize_samples, write_benchmark_report
from common.he_backend import create_context, encrypt_vector, get_backend_name, matmul


def _parse_dims(raw: str) -> list[tuple[int, int]]:
    dims = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        left, right = token.lower().split("x", maxsplit=1)
        dims.append((int(left), int(right)))
    if not dims:
        raise ValueError("No dimensions provided")
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark backend matmul using common.he_backend")
    parser.add_argument("--samples", type=int, default=3, help="Number of measured samples per dimension.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument(
        "--dims",
        type=str,
        default="256x128,512x256",
        help="Comma-separated matrix sizes as D_inxD_out.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="client/config/client_config.yaml",
        help="Client config path used to create the HE context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_he_matmul_backend.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--poly-modulus-degree",
        type=int,
        default=None,
        help="Optional override for poly_modulus_degree.",
    )
    parser.add_argument(
        "--coeff-mod-bit-sizes",
        type=str,
        default=None,
        help="Optional comma-separated coeff modulus bit sizes override.",
    )
    parser.add_argument(
        "--global-scale",
        type=int,
        default=None,
        help="Optional global scale override.",
    )
    args = parser.parse_args()

    ckks = load_ckks_params(Path(args.config_path))
    if args.poly_modulus_degree is not None:
        ckks["poly_modulus_degree"] = int(args.poly_modulus_degree)
    if args.coeff_mod_bit_sizes is not None:
        ckks["coeff_mod_bit_sizes"] = [
            int(part.strip()) for part in args.coeff_mod_bit_sizes.split(",") if part.strip()
        ]
    if args.global_scale is not None:
        ckks["global_scale"] = int(args.global_scale)

    context = create_context(
        poly_modulus_degree=int(ckks["poly_modulus_degree"]),
        coeff_mod_bit_sizes=list(ckks["coeff_mod_bit_sizes"]),
        global_scale=int(ckks["global_scale"]),
        use_galois_keys=bool(ckks.get("use_galois_keys", True)),
        use_relin_keys=bool(ckks.get("use_relin_keys", True)),
    )

    rng = seeded_rng(args.seed)
    dims = _parse_dims(args.dims)
    results = []

    for d_in, d_out in dims:
        vector = rng.normal(0.0, 0.01, size=d_in).astype(np.float32)
        weight = rng.normal(0.0, 0.01, size=(d_in, d_out)).astype(np.float32)

        samples = benchmark_operation(
            lambda: matmul(encrypt_vector(context, vector), weight),
            samples=args.samples,
            warmups=args.warmups,
        )
        results.append(summarize_samples(f"he_backend_matmul_{d_in}x{d_out}", samples))

    output_path = write_benchmark_report(
        args.output,
        results,
        metadata={"matrix_source": "synthetic", "seed": args.seed, "backend": get_backend_name()},
        ckks_params=ckks,
    )

    print(f"Wrote {output_path}")
    for result in results:
        print(f"{result['name']}: mean={result['mean_ms']}ms std={result['std_ms']}ms")


if __name__ == "__main__":
    main()
