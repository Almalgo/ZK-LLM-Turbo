"""Generate final support/no-support decision for near-term Phase 3 change."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _decision(phase3_gate: dict) -> dict:
    gate_decision = phase3_gate.get("decision", {}).get("decision")
    reasons = phase3_gate.get("decision", {}).get("reasons", [])
    supports_change = gate_decision == "go"
    return {
        "supports_change": supports_change,
        "phase3_gate_decision": gate_decision,
        "reasons": reasons,
    }


def _write_markdown(path: Path, decision: dict) -> None:
    lines = [
        "# T3 Change Decision",
        "",
        f"Date: {datetime.now(UTC).date().isoformat()}",
        "",
        f"Decision: `{'support_change' if decision['supports_change'] else 'no_support_change'}`",
        "",
        f"Phase 3 gate decision: `{decision['phase3_gate_decision']}`",
        "",
    ]
    if decision["reasons"]:
        lines.append("## Reasons")
        lines.extend([f"- {reason}" for reason in decision["reasons"]])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final T3 support/no-support decision")
    parser.add_argument(
        "--phase3-gate",
        type=Path,
        default=Path("benchmarks/results/t3_phase3_gate.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/t3_change_decision.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/T3-CHANGE-DECISION.md"),
    )
    args = parser.parse_args()

    phase3_gate = _load(args.phase3_gate)
    decision = _decision(phase3_gate)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase3_gate": phase3_gate,
        "decision": decision,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(args.output_md, decision)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
