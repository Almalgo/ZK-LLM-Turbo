"""Compare matmul correctness/latency across HE backends on small dimensions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import seeded_rng
from common.he_backend import create_context, decrypt_vector, encrypt_vector, matmul


def _parse_dims(raw: str) -> list[tuple[int, int]]:
    dims = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        d_in, d_out = token.lower().split("x", maxsplit=1)
        dims.append((int(d_in), int(d_out)))
    if not dims:
        raise ValueError("No dimensions provided")
    return dims


def _run_backend(
    backend: str,
    dims: list[tuple[int, int]],
    seed: int,
    samples: int,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: list[int],
    global_scale: int,
) -> dict:
    old_backend = os.environ.get("ZKLLM_HE_BACKEND")
    os.environ["ZKLLM_HE_BACKEND"] = backend
    rng = seeded_rng(seed)
    rows = []
    try:
        context = create_context(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale=global_scale,
            use_galois_keys=True,
            use_relin_keys=True,
        )
        for d_in, d_out in dims:
            x = rng.normal(0.0, 0.01, size=d_in).astype(np.float32)
            w = rng.normal(0.0, 0.01, size=(d_in, d_out)).astype(np.float32)
            expected = x @ w

            elapsed_samples = []
            actual = None
            for _ in range(samples):
                enc = encrypt_vector(context, x)
                t0 = time.perf_counter()
                out_enc = matmul(enc, w)
                elapsed_samples.append((time.perf_counter() - t0) * 1000.0)
                if actual is None:
                    actual = np.array(decrypt_vector(out_enc)[:d_out], dtype=np.float32)

            abs_err = np.abs(actual - expected)
            rows.append(
                {
                    "name": f"{d_in}x{d_out}",
                    "mean_ms": round(float(sum(elapsed_samples) / len(elapsed_samples)), 3),
                    "samples": int(samples),
                    "mae": float(np.mean(abs_err)),
                    "max_abs_error": float(np.max(abs_err)),
                }
            )
        return {"backend": backend, "ok": True, "results": rows, "error": None}
    except Exception as exc:
        return {"backend": backend, "ok": False, "results": rows, "error": str(exc)}
    finally:
        if old_backend is None:
            os.environ.pop("ZKLLM_HE_BACKEND", None)
        else:
            os.environ["ZKLLM_HE_BACKEND"] = old_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HE backend matmul on small matrices")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed")
    parser.add_argument("--samples", type=int, default=1, help="Number of runs per dimension.")
    parser.add_argument("--dims", type=str, default="8x4", help="Comma-separated D_inxD_out list")
    parser.add_argument(
        "--backends",
        type=str,
        default="tenseal,openfhe",
        help="Comma-separated backend names",
    )
    parser.add_argument("--poly-modulus-degree", type=int, default=16384)
    parser.add_argument("--coeff-mod-bit-sizes", type=str, default="60,40,40,60")
    parser.add_argument("--global-scale", type=int, default=2**40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/compare_he_backend_matmul.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    dims = _parse_dims(args.dims)
    coeff_mod_bit_sizes = [int(part.strip()) for part in args.coeff_mod_bit_sizes.split(",") if part.strip()]
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dims": [{"d_in": d_in, "d_out": d_out} for d_in, d_out in dims],
        "params": {
            "poly_modulus_degree": args.poly_modulus_degree,
            "coeff_mod_bit_sizes": coeff_mod_bit_sizes,
            "global_scale": args.global_scale,
        },
        "backends": [
            _run_backend(
                backend=backend,
                dims=dims,
                seed=args.seed,
                samples=args.samples,
                poly_modulus_degree=args.poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
                global_scale=args.global_scale,
            )
            for backend in backends
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
