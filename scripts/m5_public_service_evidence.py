#!/usr/bin/env python3
"""Capture public Milestone 5 SNET service evidence.

This script is intentionally stricter than a plain HTTP status check. The HaaS
daemon heartbeat can return HTTP 200 while still reporting the service as
Offline/NOT_SERVING, which is not acceptable reviewer-facing evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import current_git_sha


DEFAULT_DAEMON_URL = (
    "https://1cf76242e956da6397b473a57f2ca4a9245568fe90b76910c1b18b4f."
    "mainnet.haas.singularitynet.dev"
)
DEFAULT_HOSTED_HEARTBEAT_URL = (
    "https://api.haas.singularitynet.dev/orchestrator/v1/services/"
    "almalgo_labs/zk_llm1/heartbeat"
)
DEFAULT_MARKETPLACE_URL = (
    "https://marketplace.singularitynet.io/servicedetails/org/"
    "almalgo_labs/service/zk_llm1/tab/0"
)


def _request(url: str, timeout: float) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "url": url,
            "checked_at_utc": started,
            "ok": False,
            "error": str(exc),
        }

    body = response.text
    parsed_json = None
    try:
        parsed_json = response.json()
    except ValueError:
        pass

    return {
        "url": url,
        "checked_at_utc": started,
        "ok": 200 <= response.status_code < 300,
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": body,
        "json": parsed_json,
    }


def _parse_nested_serviceheartbeat(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("serviceheartbeat")
    if not isinstance(raw, str):
        return None
    try:
        nested = json.loads(raw)
    except ValueError:
        return None
    return nested if isinstance(nested, dict) else None


def _evaluate_daemon(check: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if check.get("status_code") != 200:
        failures.append(f"daemon heartbeat expected HTTP 200, got {check.get('status_code')}")

    payload = check.get("json")
    if isinstance(payload, dict):
        daemon_status = payload.get("status")
        if isinstance(daemon_status, str) and daemon_status.lower() == "offline":
            failures.append("daemon heartbeat reports status=Offline")

        serviceheartbeat = _parse_nested_serviceheartbeat(payload)
        if serviceheartbeat:
            service_status = serviceheartbeat.get("status")
            if isinstance(service_status, str) and service_status.upper() == "NOT_SERVING":
                failures.append("daemon serviceheartbeat reports status=NOT_SERVING")
    else:
        failures.append("daemon heartbeat did not return JSON")

    return (not failures), failures


def _evaluate_hosted(check: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if check.get("status_code") != 200:
        failures.append(f"hosted heartbeat expected HTTP 200, got {check.get('status_code')}")

    payload = check.get("json")
    if isinstance(payload, dict):
        if payload.get("status") == 503:
            failures.append("hosted heartbeat reports endpoint unhealthy")
        if payload.get("error"):
            failures.append(f"hosted heartbeat error: {payload.get('error')}")

    return (not failures), failures


def _write_report(output: Path, report: dict[str, Any]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture public Milestone 5 SNET evidence")
    parser.add_argument("--daemon-base-url", default=DEFAULT_DAEMON_URL)
    parser.add_argument("--hosted-heartbeat-url", default=DEFAULT_HOSTED_HEARTBEAT_URL)
    parser.add_argument("--marketplace-url", default=DEFAULT_MARKETPLACE_URL)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m5_public_service_evidence.json"),
    )
    args = parser.parse_args()

    daemon_base_url = args.daemon_base_url.rstrip("/")
    daemon_heartbeat_url = f"{daemon_base_url}/heartbeat"

    daemon_check = _request(daemon_heartbeat_url, args.timeout)
    hosted_check = _request(args.hosted_heartbeat_url, args.timeout)
    marketplace_check = _request(args.marketplace_url, args.timeout)

    daemon_ok, daemon_failures = _evaluate_daemon(daemon_check)
    hosted_ok, hosted_failures = _evaluate_hosted(hosted_check)

    failures = daemon_failures + hosted_failures
    evidence_ok = daemon_ok and hosted_ok

    report = {
        "name": "m5_public_service_evidence",
        "status": "pass" if evidence_ok else "fail",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "service": {
            "organization_id": "almalgo_labs",
            "service_id": "zk_llm1",
            "daemon_id": "2b4d4f3b-c043-4671-ad6a-697012038574",
            "hosted_service_id": "cb4c9cfb-bbf9-49b2-8357-d18926a7a5df",
            "daemon_base_url": daemon_base_url,
            "hosted_heartbeat_url": args.hosted_heartbeat_url,
            "marketplace_url": args.marketplace_url,
        },
        "acceptance": {
            "daemon_heartbeat_serving": daemon_ok,
            "hosted_heartbeat_healthy": hosted_ok,
            "failures": failures,
        },
        "checks": {
            "daemon_heartbeat": daemon_check,
            "hosted_heartbeat": hosted_check,
            "marketplace_url": marketplace_check,
        },
    }
    written = _write_report(args.output, report)

    print(f"{'PASS' if evidence_ok else 'FAIL'} wrote {written}")
    for failure in failures:
        print(f"FAIL {failure}")

    return 0 if evidence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
