#!/usr/bin/env python3
"""Probe OpenFHE backend readiness and smoke operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.openfhe_probe import probe_openfhe_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenFHE backend readiness")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON result",
    )
    args = parser.parse_args()

    probe = probe_openfhe_backend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(probe, indent=2) + "\n")
        print(f"Wrote {args.output}")

    if args.json:
        print(json.dumps(probe, indent=2))
        return

    print(f"openfhe_available={probe['openfhe_available']} probe_passed={probe['probe_passed']}")
    if probe.get("error"):
        print(f"error={probe['error']}")


if __name__ == "__main__":
    main()
