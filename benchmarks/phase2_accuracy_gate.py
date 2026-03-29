"""Produce a lightweight Phase 2 accuracy gate report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.inference.nonlinear_ops import poly_silu, silu
from common.constants import COEFF_MOD_BIT_SIZES, GLOBAL_SCALE, POLY_MODULUS_DEGREE
from server.inference.he_ops import HE_POLY_SILU_COEFFS, compute_ffn_merged


def _make_context() -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def _he_poly_silu(x: np.ndarray) -> np.ndarray:
    c0, c1, c2 = HE_POLY_SILU_COEFFS
    return (c0 + c1 * x + c2 * np.square(x)).astype(np.float32)


def _activation_metrics() -> dict:
    x = np.linspace(-5.0, 5.0, 2001, dtype=np.float32)
    exact = silu(x)
    reference = poly_silu(x)
    he_safe = _he_poly_silu(x)

    def summarize(name: str, approx: np.ndarray) -> dict:
        err = approx - exact
        return {
            "name": name,
            "mae": float(np.mean(np.abs(err))),
            "max_abs_error": float(np.max(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
        }

    return {
        "reference_degree6": summarize("reference_degree6", reference),
        "he_quadratic": summarize("he_quadratic", he_safe),
    }


def _merged_ffn_metrics(samples: int = 3) -> dict:
    rng = np.random.default_rng(0)
    ctx = _make_context()
    errors = []

    for _ in range(samples):
        dim_in = 8
        dim_ffn = 12
        dim_out = 6
        x = rng.normal(0.0, 0.1, size=dim_in).astype(np.float32)
        gate_w = rng.normal(0.0, 0.1, size=(dim_in, dim_ffn)).astype(np.float32)
        up_w = rng.normal(0.0, 0.1, size=(dim_in, dim_ffn)).astype(np.float32)
        down_w = rng.normal(0.0, 0.1, size=(dim_ffn, dim_out)).astype(np.float32)

        gate = x @ gate_w
        up = x @ up_w
        expected = (_he_poly_silu(gate) * up) @ down_w

        enc_x = ts.ckks_vector(ctx, x.tolist())
        actual_enc = compute_ffn_merged(
            enc_x,
            {
                "gate_proj": gate_w,
                "up_proj": up_w,
                "down_proj": down_w,
            },
            [dim_ffn],
        )
        actual = np.array(actual_enc.decrypt()[:dim_out], dtype=np.float32)
        err = actual - expected
        errors.append(
            {
                "mae": float(np.mean(np.abs(err))),
                "max_abs_error": float(np.max(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
            }
        )

    return {
        "samples": samples,
        "mean_mae": float(np.mean([sample["mae"] for sample in errors])),
        "mean_max_abs_error": float(np.mean([sample["max_abs_error"] for sample in errors])),
        "mean_rmse": float(np.mean([sample["rmse"] for sample in errors])),
    }


def main() -> None:
    output_path = Path("benchmarks/results/phase2_accuracy_gate.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "ckks": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "global_scale": GLOBAL_SCALE,
        },
        "activation": _activation_metrics(),
        "merged_ffn_small_dim": _merged_ffn_metrics(),
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
