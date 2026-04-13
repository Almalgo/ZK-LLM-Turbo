"""Generate a concise Phase 3 readiness markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_decision(path: Path) -> tuple[str, list[str]]:
    payload = json.loads(path.read_text())
    decision = payload.get("decision", {}).get("decision", "unknown")
    reasons = payload.get("decision", {}).get("reasons", [])
    return decision, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 readiness markdown summary")
    parser.add_argument(
        "--openfhe",
        type=Path,
        default=Path("benchmarks/results/t3_openfhe_matmul_readiness.json"),
    )
    parser.add_argument(
        "--gpu",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_readiness.json"),
    )
    parser.add_argument(
        "--poly",
        type=Path,
        default=Path("benchmarks/results/t3_polynomial_readiness.json"),
    )
    parser.add_argument(
        "--noninteractive",
        type=Path,
        default=Path("benchmarks/results/t3_noninteractive_readiness.json"),
    )
    parser.add_argument(
        "--phase3-gate",
        type=Path,
        default=Path("benchmarks/results/t3_phase3_gate.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/T3-PHASE3-READINESS-SUMMARY.md"),
    )
    args = parser.parse_args()

    rows = [
        ("T3.1 OpenFHE", args.openfhe),
        ("T3.3 GPU", args.gpu),
        ("T3.4 Polynomial", args.poly),
        ("T3.5 Non-Interactive", args.noninteractive),
        ("T3.6 Phase Gate", args.phase3_gate),
    ]

    lines = ["# Phase 3 Readiness Summary", "", "| Area | Decision |", "|---|---|"]
    details = []
    for label, path in rows:
        decision, reasons = _load_decision(path)
        lines.append(f"| {label} | `{decision}` |")
        if reasons:
            details.append(f"## {label} reasons")
            details.extend([f"- {reason}" for reason in reasons])
            details.append("")

    lines.append("")
    lines.extend(details)
    args.output.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
