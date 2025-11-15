import tenseal as ts
import base64

def decrypt_payload(context: ts.Context, enc_b64: str):
    """Decrypt a single base64 ciphertext (for testing)."""
    data = base64.b64decode(enc_b64)
    enc_vec = ts.ckks_vector_from(context, data)
    return enc_vec.decrypt()
