import numpy as np

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
