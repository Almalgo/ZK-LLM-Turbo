"""Build a T2.1 Intel HEXL acceptance report from benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.hexl_probe import probe_hexl_linkage


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _results_by_name(report: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in report.get("results", [])}


def _compare_reports(baseline: dict, candidate: dict) -> list[dict]:
    baseline_by_name = _results_by_name(baseline)
    candidate_by_name = _results_by_name(candidate)
    names = sorted(set(baseline_by_name) & set(candidate_by_name))
    comparisons = []
    for name in names:
        base_mean = float(baseline_by_name[name]["mean_ms"])
        cand_mean = float(candidate_by_name[name]["mean_ms"])
        if base_mean <= 0:
            speedup = 0.0
            improvement_pct = 0.0
        else:
            speedup = base_mean / cand_mean
            improvement_pct = ((base_mean - cand_mean) / base_mean) * 100.0
        comparisons.append(
            {
                "name": name,
                "baseline_mean_ms": round(base_mean, 3),
                "candidate_mean_ms": round(cand_mean, 3),
                "speedup": round(speedup, 4),
                "improvement_pct": round(improvement_pct, 3),
            }
        )
    return comparisons


def _acceptance_decision(
    probe: dict,
    comparisons: list[dict],
    min_required_speedup: float,
) -> dict:
    avg_speedup = 0.0
    if comparisons:
        avg_speedup = sum(item["speedup"] for item in comparisons) / len(comparisons)

    linked = bool(probe.get("hexl_linked", False))
    speedup_met = bool(comparisons) and avg_speedup >= min_required_speedup
    accepted = linked and speedup_met

    reasons = []
    if not linked:
        reasons.append("HEXL linkage not detected in active TenSEAL binaries.")
    if not comparisons:
        reasons.append("No overlapping benchmark operation names found for comparison.")
    elif not speedup_met:
        reasons.append(
            f"Average speedup {avg_speedup:.3f} is below required threshold {min_required_speedup:.3f}."
        )

    return {
        "accepted": accepted,
        "average_speedup": round(avg_speedup, 4),
        "min_required_speedup": min_required_speedup,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate T2.1 Intel HEXL acceptance report.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline bench_he_matmul report JSON path.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate HEXL bench_he_matmul report JSON path.")
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=None,
        help="Optional path to cached probe_hexl_linkage --json output. If omitted, probe live environment.",
    )
    parser.add_argument(
        "--min-required-speedup",
        type=float,
        default=1.05,
        help="Minimum average speedup required for acceptance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/hexl_acceptance_report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    baseline = _load_report(args.baseline)
    candidate = _load_report(args.candidate)
    comparisons = _compare_reports(baseline, candidate)

    if args.probe_json is not None:
        probe = json.loads(args.probe_json.read_text())
    else:
        probe = probe_hexl_linkage()

    decision = _acceptance_decision(
        probe=probe,
        comparisons=comparisons,
        min_required_speedup=args.min_required_speedup,
    )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "baseline_report": str(args.baseline),
        "candidate_report": str(args.candidate),
        "probe": probe,
        "comparisons": comparisons,
        "decision": decision,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"accepted={decision['accepted']} avg_speedup={decision['average_speedup']}")


if __name__ == "__main__":
    main()
