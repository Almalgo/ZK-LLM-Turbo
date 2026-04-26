from common.he_backend import encrypt_vector
from client.encryption.ckks_context import create_ckks_context


def encrypt_embeddings(embeddings_np, context):
    """Encrypt a list of embedding vectors."""
    if context is None:
        context = create_ckks_context()

    encrypted_vectors = []
    for vec in embeddings_np:
        enc_vec = encrypt_vector(context, vec.tolist())
        encrypted_vectors.append(enc_vec)
    return encrypted_vectors
