import pytest
from client.model.embedding_extractor import extract_embeddings, load_model, get_model_components
from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt


@pytest.mark.slow
def test_extract_embeddings_shape():
    tokenizer = load_tokenizer()
    tokens = tokenize_prompt("Test sentence", tokenizer)
    embeddings = extract_embeddings(tokens["input_ids"])
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 2048  # TinyLlama hidden dim


@pytest.mark.slow
def test_model_caching():
    """Model should be loaded once and cached."""
    m1 = load_model()
    m2 = load_model()
    assert m1 is m2


@pytest.mark.slow
def test_model_components():
    """get_model_components should return required keys."""
    components = get_model_components()
    assert "model" in components
    assert "layers" in components
    assert "final_norm_weight" in components
    assert "lm_head_weight" in components
    assert "config" in components
    assert len(components["layers"]) == 22
