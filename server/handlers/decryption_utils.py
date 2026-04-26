import base64
from common.he_backend import decrypt_vector, vector_from_bytes


def decrypt_payload(context, enc_b64: str):
    """Decrypt a single base64 ciphertext (for testing)."""
    if context is None:
        return []

    data = base64.b64decode(enc_b64)
    try:
        enc_vec = vector_from_bytes(context, data)
        return decrypt_vector(enc_vec)
    except Exception:
        return []
