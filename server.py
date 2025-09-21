from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import tenseal as ts
import numpy as np

app = FastAPI()

# --- Request Schema ---
class EmbeddingRequest(BaseModel):
    encrypted_embeddings: list[str]  # Base64-encoded byte strings
    metadata: dict

# --- Dummy Response (Encrypted) ---
@app.post("/api/infer")
async def infer(payload: EmbeddingRequest):
    # Load context (must match client)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Deserialize and decrypt embeddings
    decrypted_embeddings = []
    for enc_b64 in payload.encrypted_embeddings:
        encrypted_bytes = base64.b64decode(enc_b64)
        enc_vec = ts.ckks_vector_from(context, encrypted_bytes)
        decrypted = enc_vec.decrypt()
        decrypted_embeddings.append(decrypted)

    print("Received", len(decrypted_embeddings), "embeddings.")
    print("Example (first 5 dims of first token):", decrypted_embeddings[0][:5])

    # Dummy encrypted response
    dummy_result = [0.1, 0.2, 0.3]
    enc_result = ts.ckks_vector(context, dummy_result)
    response_payload = base64.b64encode(enc_result.serialize()).decode("utf-8")

    return {"encrypted_result": response_payload}
