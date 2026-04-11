#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.hexl_probe import probe_hexl_linkage


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe whether current TenSEAL binaries appear HEXL-linked.")
    parser.add_argument(
        "--require-linked",
        action="store_true",
        help="Exit with code 1 if HEXL linkage is not detected.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON probe output.",
    )
    args = parser.parse_args()

    probe = probe_hexl_linkage()

    if args.json:
        print(json.dumps(probe, indent=2))
    else:
        print(f"avx512_detected={probe['avx512_detected']}")
        print(f"hexl_linked={probe['hexl_linked']}")
        if probe["linked_binaries"]:
            print("linked_binaries:")
            for path in probe["linked_binaries"]:
                print(f"  - {path}")
        elif probe["probed_binaries"]:
            print("probed_binaries:")
            for path in probe["probed_binaries"]:
                print(f"  - {path}")
        else:
            print("No TenSEAL native binaries were discovered in the active Python environment.")

    if args.require_linked and not probe["hexl_linked"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
