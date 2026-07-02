#!/usr/bin/env python3
"""Preflight SNET hosted service and daemon endpoints before Publisher updates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from urllib.parse import urlparse

import requests


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _validate_base_url(name: str, base_url: str, *, allow_http: bool) -> list[CheckResult]:
    parsed = urlparse(base_url)
    results = []

    if parsed.scheme not in {"http", "https"}:
        results.append(CheckResult(name, False, "URL must start with http:// or https://"))
    elif parsed.scheme == "http" and not allow_http:
        results.append(CheckResult(name, False, "public SNET endpoints should use https://"))
    else:
        results.append(CheckResult(name, True, "scheme ok"))

    host = parsed.netloc.lower()
    invalid_hosts = ("github.com", "api.haas.singularitynet.dev")
    if any(bad in host for bad in invalid_hosts):
        results.append(
            CheckResult(
                name,
                False,
                "URL points to GitHub or the SNET HaaS orchestrator, not a running FastAPI service",
            )
        )
    elif host.startswith(("localhost", "127.0.0.1")) and not allow_http:
        results.append(CheckResult(name, False, "URL is local-only and cannot be reached by HaaS"))
    elif host.startswith(("localhost", "127.0.0.1")):
        results.append(CheckResult(name, True, "local host allowed for local/private preflight"))
    else:
        results.append(CheckResult(name, True, "host ok"))

    if parsed.path not in {"", "/"}:
        results.append(CheckResult(name, False, "base URL must not include a path suffix"))
    else:
        results.append(CheckResult(name, True, "base path ok"))

    return results


def _probe(
    url: str,
    timeout: float,
    *,
    method: str = "GET",
    json_body: dict | None = None,
) -> CheckResult:
    try:
        response = requests.request(method, url, json=json_body, timeout=timeout)
    except requests.RequestException as exc:
        return CheckResult(f"{method} {url}", False, str(exc))

    if response.status_code != 200:
        return CheckResult(
            f"{method} {url}",
            False,
            f"expected HTTP 200, got {response.status_code}",
        )

    return CheckResult(f"{method} {url}", True, "HTTP 200")


def _print_results(results: list[CheckResult]) -> bool:
    ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {result.name}: {result.detail}")
        ok = ok and result.ok
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SNET hosted service and optional daemon endpoints."
    )
    parser.add_argument("--public-base-url", required=True, help="Hosted service/FastAPI base URL")
    parser.add_argument("--daemon-base-url", help="Optional HaaS daemon base URL")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-http",
        action="store_true",
        help="Allow http:// URLs for local/private preflight only",
    )
    args = parser.parse_args()

    public_base_url = args.public_base_url.rstrip("/")
    results = _validate_base_url("public-base-url", public_base_url, allow_http=args.allow_http)
    results.append(_probe(f"{public_base_url}/", args.timeout))
    results.append(_probe(f"{public_base_url}/heartbeat", args.timeout))
    results.append(_probe(f"{public_base_url}/health", args.timeout))
    results.append(
        _probe(
            f"{public_base_url}/health",
            args.timeout,
            method="POST",
            json_body={"op": "health"},
        )
    )

    if args.daemon_base_url:
        daemon_base_url = args.daemon_base_url.rstrip("/")
        results.extend(
            _validate_base_url("daemon-base-url", daemon_base_url, allow_http=args.allow_http)
        )
        results.append(_probe(f"{daemon_base_url}/heartbeat", args.timeout))

    return 0 if _print_results(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
