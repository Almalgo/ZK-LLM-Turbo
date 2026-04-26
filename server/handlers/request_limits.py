"""Shared size/shape limits for layer/session request handling."""

from __future__ import annotations

MAX_PUBLIC_CONTEXT_B64_CHARS = 1_200_000
MAX_PUBLIC_CONTEXT_BYTES = 2_000_000

MAX_LAYER_JSON_VECTORS = 256
MAX_LAYER_BINARY_VECTOR_BYTES = 10_000_000
MAX_LAYER_BINARY_BODY_BYTES = 10_000_000

SUPPORTED_OPERATIONS = {
    "qkv",
    "o_proj",
    "ffn_gate_up",
    "ffn_down",
    "ffn_merged",
}

MAX_SESSION_ID_LENGTH = 128
