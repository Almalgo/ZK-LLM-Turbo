"""Generate a T3.4 full-polynomial-model readiness artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _scan_polynomial_features(repo_root: Path) -> dict:
    nonlinear_ops = (repo_root / "client" / "inference" / "nonlinear_ops.py").read_text()
    he_ops = (repo_root / "server" / "inference" / "he_ops.py").read_text()

    feature_flags = {
        "client_poly_silu": "def poly_silu(" in nonlinear_ops,
        "server_poly_silu": "def poly_silu(" in he_ops,
        "client_poly_rmsnorm": "poly_rms" in nonlinear_ops,
        "server_poly_rmsnorm": "poly_rms" in he_ops,
        "client_poly_softmax": "poly_softmax" in nonlinear_ops,
        "server_poly_softmax": "poly_softmax" in he_ops,
    }
    return feature_flags


def _decision(features: dict) -> dict:
    reasons: list[str] = []

    if not features.get("client_poly_silu") or not features.get("server_poly_silu"):
        reasons.append("SiLU polynomial coverage is incomplete.")
    if not features.get("server_poly_rmsnorm"):
        reasons.append("Server-side RMSNorm polynomial replacement is missing.")
    if not features.get("server_poly_softmax"):
        reasons.append("Server-side softmax polynomial replacement is missing.")

    ready = not reasons
    return {
        "decision": "go" if ready else "no_go",
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T3.4 polynomial readiness report")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_polynomial_readiness.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    features = _scan_polynomial_features(args.repo_root)
    decision = _decision(features)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "features": features,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']}")


if __name__ == "__main__":
    main()
