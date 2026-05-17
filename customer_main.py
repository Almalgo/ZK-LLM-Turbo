"""SingularityNET Full-Stack HaaS entrypoint.

The HaaS RunPod wrapper imports this module and calls run(input_data).
Keep the health path lightweight so deployment profiling does not load the
model or require encrypted inference fixtures.
"""

from __future__ import annotations

from fastapi import HTTPException

from server.model.weight_manager import ModelUnavailableError, get_model_status
from server.services.layer_service import process_layer_request
from server.services.session_service import create_session_from_public_context

SERVICE_NAME = "zk-llm-turbo"
LAYER_OPERATIONS = {"qkv", "o_proj", "ffn_gate_up", "ffn_down", "ffn_merged"}


def _error(message: str, error_type: str = "InvalidInput") -> dict:
    return {"error": message, "error_type": error_type}


def _health() -> dict:
    return {"status": "ok", "service": SERVICE_NAME, **get_model_status()}


def _session(input_data: dict) -> dict:
    public_context_b64 = input_data.get("public_context_b64")
    if not isinstance(public_context_b64, str) or not public_context_b64:
        return _error("public_context_b64 is required")

    try:
        return create_session_from_public_context(public_context_b64)
    except Exception as exc:
        return _error(f"Invalid public context: {exc}", type(exc).__name__)


def _layer(input_data: dict) -> dict:
    try:
        return process_layer_request(input_data)
    except ModelUnavailableError as exc:
        return _error(f"Model is not available: {exc}", "ModelUnavailableError")
    except HTTPException as exc:
        return _error(str(exc.detail), "HTTPException")
    except Exception as exc:
        return _error(f"Inference error: {exc}", type(exc).__name__)


def run(input_data):
    if not isinstance(input_data, dict):
        return _error("input_data must be an object")

    op = input_data.get("op") or input_data.get("operation")
    if op is None and "public_context_b64" in input_data:
        op = "session"
    if op in LAYER_OPERATIONS:
        op = "layer"

    if op == "health":
        return _health()
    if op == "session":
        return _session(input_data)
    if op == "layer":
        return _layer(input_data)

    if op is None:
        return _error("op is required")
    return _error(f"Unsupported op: {op}")
