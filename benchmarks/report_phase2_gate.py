"""Build a T2.6 Phase 2 gate decision report from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _prompt_aggregate(report: dict, mode: str) -> dict:
    prompt = report.get("prompt_mode_comparison", {})
    aggregate = prompt.get("aggregate", {})
    return aggregate.get(mode, {})


def _build_summary(base_report: dict, merged_report: dict, poly_report: dict | None) -> dict:
    ref = base_report.get("activation", {}).get("reference_degree6", {})
    he_quad = base_report.get("activation", {}).get("he_quadratic", {})
    merged_small = base_report.get("merged_ffn_small_dim", {})
    merged_prompt = _prompt_aggregate(merged_report, "merged_he")
    poly_prompt = _prompt_aggregate(poly_report or {}, "poly_split")

    ref_mae = float(ref.get("mae", 0.0))
    he_mae = float(he_quad.get("mae", 0.0))
    mae_ratio = 0.0 if ref_mae <= 0 else he_mae / ref_mae

    return {
        "activation": {
            "reference_degree6_mae": ref_mae,
            "he_quadratic_mae": he_mae,
            "he_vs_reference_mae_ratio": mae_ratio,
            "reference_degree6_max_abs_error": float(ref.get("max_abs_error", 0.0)),
            "he_quadratic_max_abs_error": float(he_quad.get("max_abs_error", 0.0)),
        },
        "merged_ffn_small_dim": {
            "samples": int(merged_small.get("samples", 0)),
            "mean_mae": float(merged_small.get("mean_mae", 0.0)),
            "mean_max_abs_error": float(merged_small.get("mean_max_abs_error", 0.0)),
        },
        "prompt_level": {
            "merged_he": {
                "exact_match_rate": float(merged_prompt.get("exact_match_rate", 0.0)),
                "mean_token_agreement": float(merged_prompt.get("mean_token_agreement", 0.0)),
            },
            "poly_split": {
                "exact_match_rate": float(poly_prompt.get("exact_match_rate", 0.0)),
                "mean_token_agreement": float(poly_prompt.get("mean_token_agreement", 0.0)),
            },
        },
        "failures": {
            "merged_he": merged_report.get("prompt_mode_comparison", {}).get("failures", []),
            "poly_split": (poly_report or {}).get("prompt_mode_comparison", {}).get("failures", []),
        },
    }


def _gate_decision(
    summary: dict,
    *,
    max_small_dim_mae: float,
    min_merged_exact_match_rate: float,
) -> dict:
    reasons: list[str] = []

    small_dim_mae = summary["merged_ffn_small_dim"]["mean_mae"]
    merged_exact = summary["prompt_level"]["merged_he"]["exact_match_rate"]
    merged_failures = summary["failures"]["merged_he"]
    poly_failures = summary["failures"]["poly_split"]

    systems_go = small_dim_mae <= max_small_dim_mae and not merged_failures and not poly_failures
    if not systems_go:
        if small_dim_mae > max_small_dim_mae:
            reasons.append(
                f"Merged FFN small-dim MAE {small_dim_mae:.3e} exceeds threshold {max_small_dim_mae:.3e}."
            )
        if merged_failures:
            reasons.append("Prompt comparison contains merged_he execution failures.")
        if poly_failures:
            reasons.append("Prompt comparison contains poly_split execution failures.")

    quality_parity_go = merged_exact >= min_merged_exact_match_rate
    if not quality_parity_go:
        reasons.append(
            "Merged HE prompt exact-match rate "
            f"{merged_exact:.3f} is below threshold {min_merged_exact_match_rate:.3f}."
        )

    if systems_go and quality_parity_go:
        decision = "go"
    elif systems_go:
        decision = "conditional_go"
    else:
        decision = "no_go"

    return {
        "decision": decision,
        "systems_go": systems_go,
        "quality_parity_go": quality_parity_go,
        "max_small_dim_mae": max_small_dim_mae,
        "min_merged_exact_match_rate": min_merged_exact_match_rate,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T2.6 Phase 2 gate decision report.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("benchmarks/results/phase2_accuracy_gate.json"),
        help="Base phase2_accuracy_gate benchmark JSON.",
    )
    parser.add_argument(
        "--live-merged",
        type=Path,
        default=Path("benchmarks/results/phase2_accuracy_gate_live_exact_vs_merged_five_short.json"),
        help="Live prompt comparison JSON for exact_split vs merged_he.",
    )
    parser.add_argument(
        "--live-poly",
        type=Path,
        default=Path("benchmarks/results/phase2_accuracy_gate_live_exact_vs_poly_five_short.json"),
        help="Live prompt comparison JSON for exact_split vs poly_split.",
    )
    parser.add_argument(
        "--max-small-dim-mae",
        type=float,
        default=1e-5,
        help="Maximum acceptable mean MAE for merged FFN small-dim parity check.",
    )
    parser.add_argument(
        "--min-merged-exact-match-rate",
        type=float,
        default=0.8,
        help="Minimum acceptable exact match rate for merged_he vs exact_split prompt check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/phase2_gate_decision_report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    base_report = _load_json(args.base)
    merged_report = _load_json(args.live_merged)
    poly_report = _load_json(args.live_poly) if args.live_poly.exists() else None

    summary = _build_summary(base_report, merged_report, poly_report)
    decision = _gate_decision(
        summary,
        max_small_dim_mae=args.max_small_dim_mae,
        min_merged_exact_match_rate=args.min_merged_exact_match_rate,
    )

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "inputs": {
            "base": str(args.base),
            "live_merged": str(args.live_merged),
            "live_poly": str(args.live_poly),
        },
        "summary": summary,
        "decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']} systems_go={decision['systems_go']} quality_parity_go={decision['quality_parity_go']}")


if __name__ == "__main__":
    main()
