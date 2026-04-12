import numpy as np
import pytest

from common.he_backend import (
    OPENFHE_AVAILABLE,
    OPENFHE_IMPORT_ERROR,
    create_context,
    decrypt_vector,
    encrypt_vector,
    get_backend_name,
    get_backend_status,
    serialize_public_context,
    serialize_vector,
    vector_from_bytes,
)


def test_tenseal_backend_roundtrip():
    context = create_context(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
        global_scale=2**40,
        use_galois_keys=True,
        use_relin_keys=True,
    )
    vec = encrypt_vector(context, [1.0, 2.0, 3.0])
    raw = serialize_vector(vec)
    restored = vector_from_bytes(context, raw)
    decrypted = np.array(decrypt_vector(restored)[:3], dtype=np.float32)
    np.testing.assert_allclose(decrypted, np.array([1.0, 2.0, 3.0], dtype=np.float32), atol=1e-3)


def test_backend_status_reports_openfhe_state():
    status = get_backend_status()
    assert status["selected_backend"] == get_backend_name()
    assert status["openfhe_available"] == OPENFHE_AVAILABLE
    assert status["openfhe_import_error"] == OPENFHE_IMPORT_ERROR
    assert "gpu_available" in status
    assert "gpu_error" in status


def test_public_context_serialization_excludes_secret_key():
    context = create_context(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 60],
        global_scale=2**40,
        use_galois_keys=True,
        use_relin_keys=True,
    )
    public_bytes = serialize_public_context(context)
    assert isinstance(public_bytes, bytes)
    assert len(public_bytes) > 0


def test_openfhe_encrypt_vector_returns_wrapped_vector(monkeypatch):
    from common import he_backend

    class FakeCiphertext:
        def __init__(self, raw: bytes):
            self.raw = raw

    class FakePlaintext:
        def __init__(self, values):
            self._values = values

        def SetLength(self, _length):
            return None

        def GetCKKSPackedValue(self):
            return [complex(v, 0.0) for v in self._values]

    class FakeContext:
        def MakeCKKSPackedPlaintext(self, values):
            return FakePlaintext(values)

        def Encrypt(self, public_key, plaintext):
            assert public_key == "pk"
            return FakeCiphertext(bytes([int(plaintext._values[0])]))

        def Decrypt(self, secret_key, ciphertext):
            assert secret_key == "sk"
            return FakePlaintext([float(ciphertext.raw[0]), 2.0, 3.0])

    class FakeOpenFHE:
        BINARY = "binary"

        @staticmethod
        def Serialize(ciphertext, _mode):
            return ciphertext.raw

        @staticmethod
        def DeserializeCiphertextString(raw, _mode):
            return FakeCiphertext(raw)

    monkeypatch.setattr(he_backend, "_openfhe", FakeOpenFHE())
    monkeypatch.setattr(he_backend, "OPENFHE_AVAILABLE", True)

    context = {
        "backend": "openfhe",
        "context": FakeContext(),
        "public_key": "pk",
        "secret_key": "sk",
    }
    encrypted = he_backend.encrypt_vector(context, [1.0, 2.0, 3.0])

    assert encrypted["backend"] == "openfhe"
    assert encrypted["length"] == 3

    raw = he_backend.serialize_vector(encrypted)
    restored = he_backend.vector_from_bytes(context, raw)
    assert restored["length"] == 3

    decrypted = he_backend.decrypt_vector(restored)
    np.testing.assert_allclose(
        np.array(decrypted, dtype=np.float32),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_openfhe_matmul_raises_not_implemented():
    from common import he_backend

    with pytest.raises(NotImplementedError):
        he_backend.matmul({"backend": "openfhe"}, [[1.0]])
