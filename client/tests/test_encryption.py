import numpy as np, base64
import tenseal as ts
from client.encryption.ckks_context import create_ckks_context

def test_ckks_encryption_roundtrip():
    context = create_ckks_context()
    data = np.random.rand(10).tolist()
    enc = ts.ckks_vector(context, data)
    serialized = base64.b64encode(enc.serialize()).decode("utf-8")
    restored = ts.ckks_vector_from(context, base64.b64decode(serialized))
    decrypted = restored.decrypt()
    assert np.allclose(data, decrypted, atol=1e-3)
