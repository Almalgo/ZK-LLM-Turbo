import pytest
import numpy as np
from client.encryption.ckks_context import create_ckks_context
from server.handlers import request_limits


def test_server_accepts_valid_payload_size():
    valid_size = 1000
    assert valid_size < request_limits.MAX_LAYER_BINARY_VECTOR_BYTES


def test_server_rejects_excessive_vector_count():
    excessive = request_limits.MAX_LAYER_JSON_VECTORS + 1
    assert excessive > request_limits.MAX_LAYER_JSON_VECTORS


def test_operation_in_supported_list():
    assert "qkv" in request_limits.SUPPORTED_OPERATIONS
    assert "ffn_merged" in request_limits.SUPPORTED_OPERATIONS


def test_session_id_length_limit():
    valid_id = "a" * 100
    assert len(valid_id) <= request_limits.MAX_SESSION_ID_LENGTH