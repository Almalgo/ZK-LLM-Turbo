"""Generate a T3.3 GPU readiness decision artifact."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.he_backend import get_backend_status


def _load_openfhe_readiness(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _load_gpu_feasibility(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _decision(backend_status: dict, openfhe_readiness: dict, gpu_feasibility: dict) -> dict:
    reasons: list[str] = []
    gpu_available = bool(backend_status.get("gpu_available", False))
    openfhe_available = bool(backend_status.get("openfhe_available", False))

    if not gpu_available:
        reasons.append("GPU runtime unavailable (`nvidia-smi` probe failed).")
    if not openfhe_available:
        reasons.append("OpenFHE backend import unavailable.")

    readiness_decision = openfhe_readiness.get("decision", {}).get("decision")
    if readiness_decision and readiness_decision != "go":
        reasons.append(
            f"OpenFHE matmul readiness is `{readiness_decision}`, not `go`."
        )

    gpu_path_usable = bool(gpu_feasibility.get("gpu_path_usable", False))
    if not gpu_path_usable:
        reasons.append("No confirmed GPU HE execution path in current OpenFHE runtime.")

    ready = gpu_available and openfhe_available and readiness_decision == "go" and gpu_path_usable
    return {
        "decision": "go" if ready else "no_go",
        "gpu_available": gpu_available,
        "openfhe_available": openfhe_available,
        "openfhe_readiness_decision": readiness_decision,
        "gpu_path_usable": gpu_path_usable,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T3.3 GPU readiness report")
    parser.add_argument(
        "--openfhe-readiness",
        type=Path,
        default=Path("benchmarks/results/t3_openfhe_matmul_readiness.json"),
        help="Path to T3.1 OpenFHE matmul readiness report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_readiness.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--gpu-feasibility",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_feasibility.json"),
        help="Path to GPU feasibility probe artifact.",
    )
    args = parser.parse_args()

    backend_status = get_backend_status()
    openfhe_readiness = _load_openfhe_readiness(args.openfhe_readiness)
    gpu_feasibility = _load_gpu_feasibility(args.gpu_feasibility)
    decision = _decision(backend_status, openfhe_readiness, gpu_feasibility)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "backend_status": backend_status,
        "openfhe_readiness": openfhe_readiness,
        "gpu_feasibility": gpu_feasibility,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"decision={decision['decision']}")


if __name__ == "__main__":
    main()
