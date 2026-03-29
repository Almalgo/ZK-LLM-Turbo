from common.he_backend import encrypt_vector


def encrypt_embeddings(embeddings_np, context):
    """Encrypt a list of embedding vectors."""
    encrypted_vectors = []
    for vec in embeddings_np:
        enc_vec = encrypt_vector(context, vec.tolist())
        encrypted_vectors.append(enc_vec)
    return encrypted_vectors
