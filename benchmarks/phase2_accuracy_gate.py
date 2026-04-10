"""Produce a lightweight Phase 2 accuracy gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import require_server
from client.client import generate, load_config
from client.inference.nonlinear_ops import poly_silu, silu
from common.constants import COEFF_MOD_BIT_SIZES, GLOBAL_SCALE, POLY_MODULUS_DEGREE
from server.inference.he_ops import HE_POLY_SILU_COEFFS, compute_ffn_merged

DEFAULT_PROMPTS = [
    "Hello world",
    "Explain homomorphic encryption in one sentence.",
    "Write a haiku about latency.",
]


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


def _token_agreement(reference: list[int], candidate: list[int]) -> float:
    if not reference:
        return 1.0 if not candidate else 0.0
    matches = sum(1 for ref, got in zip(reference, candidate) if ref == got)
    return matches / len(reference)


def _prompt_mode_metrics(
    prompts: list[str],
    num_tokens: int,
    num_encrypted_layers: int,
    transport: str,
    pipeline_mode: str,
    comparison_modes: list[str] | None = None,
) -> dict:
    _, server_cfg = load_config()
    require_server(server_cfg["base_url"])

    use_websocket_override = None
    if transport == "http":
        use_websocket_override = False
    elif transport == "websocket":
        use_websocket_override = True

    use_async_pipeline_override = None
    if pipeline_mode == "sync":
        use_async_pipeline_override = False
    elif pipeline_mode == "async":
        use_async_pipeline_override = True

    modes = {
        "exact_split": {
            "use_merged_ffn_override": False,
            "use_poly_silu_override": False,
        },
        "poly_split": {
            "use_merged_ffn_override": False,
            "use_poly_silu_override": True,
        },
        "merged_he": {
            "use_merged_ffn_override": True,
            "use_poly_silu_override": True,
        },
    }

    selected_modes = comparison_modes or list(modes.keys())
    if "exact_split" not in selected_modes:
        raise ValueError("comparison_modes must include 'exact_split' as baseline")

    prompt_results = []
    mode_agreements = {mode: [] for mode in selected_modes if mode != "exact_split"}
    failures = []

    for prompt in prompts:
        runs = {}
        for mode_name in selected_modes:
            overrides = modes[mode_name]
            try:
                runs[mode_name] = generate(
                    prompt=prompt,
                    num_tokens=num_tokens,
                    num_encrypted_layers=num_encrypted_layers,
                    return_stats=True,
                    quiet=True,
                    use_websocket_override=use_websocket_override,
                    use_async_pipeline_override=use_async_pipeline_override,
                    **overrides,
                )
            except Exception as exc:
                failures.append(
                    {
                        "prompt": prompt,
                        "mode": mode_name,
                        "error": str(exc),
                    }
                )
                break

        if "exact_split" not in runs:
            continue

        baseline = runs["exact_split"]
        comparisons = {}
        for mode_name, run in runs.items():
            if mode_name == "exact_split":
                continue
            agreement = _token_agreement(
                baseline.generated_token_ids,
                run.generated_token_ids,
            )
            mode_agreements[mode_name].append(agreement)
            comparisons[mode_name] = {
                "token_agreement": agreement,
                "exact_match": run.generated_token_ids == baseline.generated_token_ids,
                "generated_token_ids": run.generated_token_ids,
                "generated_text": run.generated_text,
            }

        prompt_results.append(
            {
                "prompt": prompt,
                "exact_split": {
                    "generated_token_ids": baseline.generated_token_ids,
                    "generated_text": baseline.generated_text,
                },
                "comparisons": comparisons,
            }
        )

    return {
        "enabled": True,
        "num_prompts": len(prompts),
        "num_tokens": num_tokens,
        "num_encrypted_layers": num_encrypted_layers,
        "transport": transport,
        "pipeline_mode": pipeline_mode,
        "comparison_modes": selected_modes,
        "prompts": prompt_results,
        "failures": failures,
        "aggregate": {
            mode: {
                "mean_token_agreement": float(np.mean(values)) if values else 0.0,
                "exact_match_rate": float(np.mean([value == 1.0 for value in values])) if values else 0.0,
            }
            for mode, values in mode_agreements.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce the Phase 2 accuracy gate report.")
    parser.add_argument(
        "--include-prompt-comparisons",
        action="store_true",
        help="Run prompt-level split/poly/merged generation comparisons against a live server.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to evaluate. Repeat to provide multiple prompts.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=3,
        help="Number of tokens to generate for prompt comparisons.",
    )
    parser.add_argument(
        "--num-encrypted-layers",
        type=int,
        default=1,
        help="Encrypted layer count for prompt comparisons.",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "http", "websocket"],
        default="auto",
        help="Transport mode for prompt comparisons.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=["auto", "sync", "async"],
        default="auto",
        help="Pipeline mode for prompt comparisons.",
    )
    parser.add_argument(
        "--comparison-modes",
        nargs="*",
        choices=["exact_split", "poly_split", "merged_he"],
        default=["exact_split", "poly_split", "merged_he"],
        help="Subset of prompt comparison modes to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/phase2_accuracy_gate.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    output_path = args.output
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

    if args.include_prompt_comparisons:
        report["prompt_mode_comparison"] = _prompt_mode_metrics(
            prompts=args.prompts or DEFAULT_PROMPTS,
            num_tokens=args.num_tokens,
            num_encrypted_layers=args.num_encrypted_layers,
            transport=args.transport,
            pipeline_mode=args.pipeline_mode,
            comparison_modes=args.comparison_modes,
        )
    else:
        report["prompt_mode_comparison"] = {
            "enabled": False,
            "reason": "Run with --include-prompt-comparisons to compare exact split, poly split, and merged HE modes on live prompts.",
        }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
