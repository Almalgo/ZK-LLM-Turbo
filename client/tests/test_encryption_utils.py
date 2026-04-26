import pytest
from client.encryption.utils import serialize_encrypted_vectors, build_payload
import base64


def test_serialize_encrypted_vectors_returns_strings():
    class MockVector:
        def serialize(self):
            return b"mock_data"

    vectors = [MockVector(), MockVector()]
    result = serialize_encrypted_vectors(vectors)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)


def test_build_payload_includes_metadata():
    serialized = ["abc", "def"]
    shape = (4, 2048)
    ckks_config = {"poly_modulus_degree": 8192}

    payload = build_payload(serialized, shape, ckks_config)

    assert payload["encrypted_embeddings"] == serialized
    assert payload["metadata"]["token_count"] == 4
    assert payload["metadata"]["hidden_dim"] == 2048
    assert payload["metadata"]["ckks"] == ckks_config