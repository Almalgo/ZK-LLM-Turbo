"""Generate a T3.5 non-interactive protocol readiness artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _decision(gpu_report: dict, poly_report: dict) -> dict:
    reasons: list[str] = []
    gpu_decision = gpu_report.get("decision", {}).get("decision")
    poly_decision = poly_report.get("decision", {}).get("decision")

    if gpu_decision != "go":
        reasons.append(f"T3.3 GPU readiness is `{gpu_decision}`.")
    if poly_decision != "go":
        reasons.append(f"T3.4 polynomial readiness is `{poly_decision}`.")

    ready = gpu_decision == "go" and poly_decision == "go"
    return {
        "decision": "go" if ready else "no_go",
        "gpu_readiness_decision": gpu_decision,
        "polynomial_readiness_decision": poly_decision,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T3.5 non-interactive readiness report")
    parser.add_argument(
        "--gpu-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_readiness.json"),
        help="Path to T3.3 GPU readiness artifact.",
    )
    parser.add_argument(
        "--polynomial-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_polynomial_readiness.json"),
        help="Path to T3.4 polynomial readiness artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_noninteractive_readiness.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    gpu_report = _load(args.gpu_readiness)
    poly_report = _load(args.polynomial_readiness)
    decision = _decision(gpu_report, poly_report)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "gpu_readiness": gpu_report,
        "polynomial_readiness": poly_report,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']}")


if __name__ == "__main__":
    main()
