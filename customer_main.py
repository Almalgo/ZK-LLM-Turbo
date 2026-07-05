"""SingularityNET Full-Stack HaaS lightweight proxy entrypoint.

The HaaS RunPod wrapper imports this module and calls run(input_data). Keep the
health path lightweight so Publisher profiling does not load TinyLlama or HE
runtime dependencies. Session and layer operations are forwarded to the
self-hosted FastAPI backend that performs the heavy work.
"""

from __future__ import annotations

import os
from urllib.parse import urljoin

import requests

SERVICE_ID = os.getenv("SNET_SERVICE_ID", "zk_llm2")
SERVICE_NAME = "zk-llm-turbo"
DEFAULT_BACKEND_BASE_URL = "https://zkllm.almalgo.com"
LAYER_OPERATIONS = {"qkv", "o_proj", "ffn_gate_up", "ffn_down", "ffn_merged"}


def _error(message: str, error_type: str = "InvalidInput", **extra) -> dict:
    response = {"error": message, "error_type": error_type}
    response.update(extra)
    return response


def _backend_base_url() -> str | None:
    value = os.getenv("ZKLLM_BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL).strip()
    return value.rstrip("/") if value else None


def _backend_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("ZKLLM_BACKEND_AUTH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _timeout_seconds() -> float:
    return _float_env("ZKLLM_BACKEND_TIMEOUT_SECONDS", 900.0)


def _health_timeout_seconds() -> float:
    return _float_env("ZKLLM_BACKEND_HEALTH_TIMEOUT_SECONDS", 10.0)


def _fail_open_health() -> bool:
    return os.getenv("ZKLLM_PROXY_FAIL_OPEN_HEALTH", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _backend_url(path: str) -> str | None:
    base_url = _backend_base_url()
    if base_url is None:
        return None
    return urljoin(f"{base_url}/", path.lstrip("/"))


def _backend_not_configured() -> dict:
    return _error(
        "ZKLLM_BACKEND_BASE_URL is required for proxy operation",
        "BackendNotConfigured",
    )


def _post_backend(path: str, payload: dict, timeout: float) -> dict:
    url = _backend_url(path)
    if url is None:
        return _backend_not_configured()

    try:
        response = requests.post(url, json=payload, headers=_backend_headers(), timeout=timeout)
    except requests.RequestException as exc:
        return _error(f"Backend request failed: {exc}", "BackendRequestError")

    if response.status_code < 200 or response.status_code >= 300:
        return _error(
            f"Backend returned HTTP {response.status_code}",
            "BackendHTTPError",
            status_code=response.status_code,
            backend_body=response.text,
        )

    try:
        return response.json()
    except ValueError:
        return _error("Backend returned non-JSON response", "BackendResponseError")


def _probe_backend_health() -> dict:
    base_url = _backend_base_url()
    if base_url is None:
        return {
            "configured": False,
            "reachable": False,
            "base_url": None,
            "status_code": None,
            "error": None,
        }

    url = _backend_url("/health")
    try:
        response = requests.get(url, headers=_backend_headers(), timeout=_health_timeout_seconds())
    except requests.RequestException as exc:
        return {
            "configured": True,
            "reachable": False,
            "base_url": base_url,
            "status_code": None,
            "error": str(exc),
        }

    return {
        "configured": True,
        "reachable": 200 <= response.status_code < 300,
        "base_url": base_url,
        "status_code": response.status_code,
        "error": None if 200 <= response.status_code < 300 else response.text,
    }


def _health() -> dict:
    return {
        "serviceID": SERVICE_ID,
        "status": "SERVING",
    }


def _proxy_health() -> dict:
    backend = _probe_backend_health()
    status = "SERVING"
    if backend["configured"] and not backend["reachable"] and not _fail_open_health():
        status = "UNAVAILABLE"
    return {
        "serviceID": SERVICE_ID,
        "service": SERVICE_NAME,
        "status": status,
        "mode": "proxy",
        "backend": backend,
    }


def _proxy_session(input_data: dict) -> dict:
    public_context_b64 = input_data.get("public_context_b64")
    if not isinstance(public_context_b64, str) or not public_context_b64:
        return _error("public_context_b64 is required")
    return _post_backend("/api/session", input_data, _timeout_seconds())


def _proxy_layer(input_data: dict) -> dict:
    return _post_backend("/api/layer", input_data, _timeout_seconds())


def run(input_data):
    if not isinstance(input_data, dict):
        return _error("input_data must be an object")

    op = input_data.get("op") or input_data.get("operation")
    if op is None and "public_context_b64" in input_data:
        op = "session"
    if op in LAYER_OPERATIONS:
        op = "layer"

    if op in {"health", "heartbeat"}:
        return _health()
    if op == "proxy_health":
        return _proxy_health()
    if op == "session":
        return _proxy_session(input_data)
    if op == "layer":
        return _proxy_layer(input_data)

    if op is None:
        return _error("op is required")
    return _error(f"Unsupported op: {op}")
