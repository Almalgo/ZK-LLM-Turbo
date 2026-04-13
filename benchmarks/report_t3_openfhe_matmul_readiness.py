"""Build a T3.1 OpenFHE matmul readiness report from comparison artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    rows: dict[str, dict] = {}
    for backend_entry in payload.get("backends", []):
        backend = backend_entry.get("backend")
        for result in backend_entry.get("results", []):
            rows[f"{backend}:{result.get('name')}"] = result
    return rows


def _join_metrics(tenseal_rows: dict[str, dict], openfhe_rows: dict[str, dict]) -> list[dict]:
    names = sorted(
        {
            key.split(":", 1)[1]
            for key in tenseal_rows.keys()
            if key.startswith("tenseal:")
        }
        & {
            key.split(":", 1)[1]
            for key in openfhe_rows.keys()
            if key.startswith("openfhe:")
        }
    )
    joined = []
    for name in names:
        tenseal = tenseal_rows[f"tenseal:{name}"]
        openfhe = openfhe_rows[f"openfhe:{name}"]
        tenseal_ms = float(tenseal["mean_ms"])
        openfhe_ms = float(openfhe["mean_ms"])
        slowdown = openfhe_ms / tenseal_ms if tenseal_ms > 0 else 0.0
        joined.append(
            {
                "name": name,
                "tenseal_mean_ms": tenseal_ms,
                "openfhe_mean_ms": openfhe_ms,
                "openfhe_slowdown_vs_tenseal": slowdown,
                "tenseal_mae": float(tenseal["mae"]),
                "openfhe_mae": float(openfhe["mae"]),
                "mae_delta": float(openfhe["mae"]) - float(tenseal["mae"]),
            }
        )
    return joined


def _decision(joined: list[dict], max_allowed_slowdown: float, max_allowed_mae: float) -> dict:
    reasons = []
    if not joined:
        return {
            "decision": "no_go",
            "reasons": ["No overlapping dimensions between TenSEAL and OpenFHE artifacts."],
        }

    worst_slowdown = max(item["openfhe_slowdown_vs_tenseal"] for item in joined)
    worst_openfhe_mae = max(item["openfhe_mae"] for item in joined)

    perf_ok = worst_slowdown <= max_allowed_slowdown
    accuracy_ok = worst_openfhe_mae <= max_allowed_mae

    if not perf_ok:
        reasons.append(
            f"Worst OpenFHE slowdown {worst_slowdown:.3f} exceeds threshold {max_allowed_slowdown:.3f}."
        )
    if not accuracy_ok:
        reasons.append(
            f"Worst OpenFHE MAE {worst_openfhe_mae:.3e} exceeds threshold {max_allowed_mae:.3e}."
        )

    return {
        "decision": "go" if perf_ok and accuracy_ok else "no_go",
        "performance_ok": perf_ok,
        "accuracy_ok": accuracy_ok,
        "worst_slowdown": worst_slowdown,
        "worst_openfhe_mae": worst_openfhe_mae,
        "max_allowed_slowdown": max_allowed_slowdown,
        "max_allowed_mae": max_allowed_mae,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T3.1 OpenFHE matmul readiness report")
    parser.add_argument(
        "--tenseal-artifacts",
        type=str,
        default=(
            "benchmarks/results/compare_he_backend_matmul_512x256.json,"
            "benchmarks/results/compare_he_backend_matmul_1024x256.json,"
            "benchmarks/results/compare_he_backend_matmul_2048x256.json,"
            "benchmarks/results/compare_he_backend_matmul_tenseal_2048x1024.json,"
            "benchmarks/results/compare_he_backend_matmul_tenseal_2048x2048.json"
        ),
        help="Comma-separated TenSEAL comparison JSON paths.",
    )
    parser.add_argument(
        "--openfhe-artifacts",
        type=str,
        default=(
            "benchmarks/results/compare_he_backend_matmul_512x256.json,"
            "benchmarks/results/compare_he_backend_matmul_1024x256.json,"
            "benchmarks/results/compare_he_backend_matmul_2048x256.json,"
            "benchmarks/results/compare_he_backend_matmul_openfhe_2048x1024.json,"
            "benchmarks/results/compare_he_backend_matmul_openfhe_2048x2048.json"
        ),
        help="Comma-separated OpenFHE comparison JSON paths.",
    )
    parser.add_argument("--max-allowed-slowdown", type=float, default=1.25)
    parser.add_argument("--max-allowed-mae", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_openfhe_matmul_readiness.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    tenseal_rows = {}
    for token in args.tenseal_artifacts.split(","):
        token = token.strip()
        if token:
            tenseal_rows.update(_load_rows(Path(token)))

    openfhe_rows = {}
    for token in args.openfhe_artifacts.split(","):
        token = token.strip()
        if token:
            openfhe_rows.update(_load_rows(Path(token)))

    joined = _join_metrics(tenseal_rows, openfhe_rows)
    decision = _decision(joined, args.max_allowed_slowdown, args.max_allowed_mae)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "joined_metrics": joined,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']}")


if __name__ == "__main__":
    main()
