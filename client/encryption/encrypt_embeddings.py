import tenseal as ts

def encrypt_embeddings(embeddings_np, context: ts.Context):
    """Encrypt a list of embedding vectors."""
    encrypted_vectors = []
    for vec in embeddings_np:
        enc_vec = ts.ckks_vector(context, vec.tolist())
        encrypted_vectors.append(enc_vec)
    return encrypted_vectors
