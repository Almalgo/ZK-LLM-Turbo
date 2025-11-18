import json
from client.encryption.utils import build_payload

def test_payload_structure():
    fake_serialized = ["abc123", "def456"]
    shape = (10, 2048)
    cfg = {"poly_modulus_degree": 8192, "global_scale": 2**40}
    payload = build_payload(fake_serialized, shape, cfg)
    parsed = json.loads(json.dumps(payload))
    assert "encrypted_embeddings" in parsed
    assert "metadata" in parsed
