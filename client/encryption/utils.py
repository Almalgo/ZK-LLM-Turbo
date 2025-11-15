import base64

def serialize_encrypted_vectors(encrypted_vectors):
    """Serialize and base64-encode encrypted vectors."""
    return [base64.b64encode(vec.serialize()).decode("utf-8") for vec in encrypted_vectors]

def build_payload(serialized_vectors, embeddings_shape, ckks_config):
    """Construct JSON payload with metadata."""
    return {
        "encrypted_embeddings": serialized_vectors,
        "metadata": {
            "token_count": embeddings_shape[0],
            "hidden_dim": embeddings_shape[1],
            "ckks": ckks_config
        }
    }
