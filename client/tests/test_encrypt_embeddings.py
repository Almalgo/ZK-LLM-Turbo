import numpy as np
from client.encryption.encrypt_embeddings import encrypt_embeddings


def test_encrypt_embeddings_returns_list():
    embeddings = np.random.randn(10, 2048).astype(np.float32)
    result = encrypt_embeddings(embeddings, None)
    assert isinstance(result, list)
    assert len(result) == 10


def test_encrypt_embeddings_single_token():
    embeddings = np.random.randn(1, 2048).astype(np.float32)
    result = encrypt_embeddings(embeddings, None)
    assert len(result) == 1


def test_encrypt_embeddings_multi_token():
    embeddings = np.random.randn(5, 2048).astype(np.float32)
    result = encrypt_embeddings(embeddings, None)
    assert len(result) == 5