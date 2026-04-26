import pytest
from server.handlers.request_limits import (
    MAX_PUBLIC_CONTEXT_B64_CHARS,
    MAX_PUBLIC_CONTEXT_BYTES,
    MAX_LAYER_JSON_VECTORS,
    MAX_LAYER_BINARY_VECTOR_BYTES,
    MAX_LAYER_BINARY_BODY_BYTES,
    SUPPORTED_OPERATIONS,
    MAX_SESSION_ID_LENGTH,
)


def test_max_public_context_b64_chars_defined():
    assert MAX_PUBLIC_CONTEXT_B64_CHARS > 0
    assert isinstance(MAX_PUBLIC_CONTEXT_B64_CHARS, int)


def test_max_public_context_bytes_defined():
    assert MAX_PUBLIC_CONTEXT_BYTES > 0


def test_max_layer_json_vectors_defined():
    assert MAX_LAYER_JSON_VECTORS > 0


def test_max_layer_binary_limits_defined():
    assert MAX_LAYER_BINARY_VECTOR_BYTES > 0
    assert MAX_LAYER_BINARY_BODY_BYTES > 0


def test_supported_operations_contains_qkv():
    assert "qkv" in SUPPORTED_OPERATIONS


def test_supported_operations_contains_ffn_merged():
    assert "ffn_merged" in SUPPORTED_OPERATIONS


def test_max_session_id_length_defined():
    assert MAX_SESSION_ID_LENGTH > 0
    assert MAX_SESSION_ID_LENGTH == 128