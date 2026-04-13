"""Generate an updated T3.6 Phase 3 gate decision artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _decision(openfhe_report: dict, gpu_report: dict, poly_report: dict, ni_report: dict) -> dict:
    reasons: list[str] = []

    openfhe_decision = openfhe_report.get("decision", {}).get("decision")
    gpu_decision = gpu_report.get("decision", {}).get("decision")
    poly_decision = poly_report.get("decision", {}).get("decision")
    ni_decision = ni_report.get("decision", {}).get("decision")

    if openfhe_decision != "go":
        reasons.append(f"T3.1 readiness is `{openfhe_decision}`.")
    if gpu_decision != "go":
        reasons.append(f"T3.3 readiness is `{gpu_decision}`.")
    if poly_decision != "go":
        reasons.append(f"T3.4 readiness is `{poly_decision}`.")
    if ni_decision != "go":
        reasons.append(f"T3.5 readiness is `{ni_decision}`.")

    go = not reasons
    return {
        "decision": "go" if go else "no_go",
        "openfhe_readiness_decision": openfhe_decision,
        "gpu_readiness_decision": gpu_decision,
        "polynomial_readiness_decision": poly_decision,
        "noninteractive_readiness_decision": ni_decision,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T3.6 Phase 3 gate report")
    parser.add_argument(
        "--openfhe-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_openfhe_matmul_readiness.json"),
    )
    parser.add_argument(
        "--gpu-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_readiness.json"),
    )
    parser.add_argument(
        "--polynomial-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_polynomial_readiness.json"),
    )
    parser.add_argument(
        "--noninteractive-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_noninteractive_readiness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_phase3_gate.json"),
    )
    args = parser.parse_args()

    openfhe_report = _load(args.openfhe_readiness)
    gpu_report = _load(args.gpu_readiness)
    poly_report = _load(args.polynomial_readiness)
    ni_report = _load(args.noninteractive_readiness)

    decision = _decision(openfhe_report, gpu_report, poly_report, ni_report)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "openfhe_readiness": openfhe_report,
        "gpu_readiness": gpu_report,
        "polynomial_readiness": poly_report,
        "noninteractive_readiness": ni_report,
        "decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']}")


if __name__ == "__main__":
    main()
