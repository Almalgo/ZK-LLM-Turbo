from client.model.embedding_extractor import extract_embeddings
from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt

def test_extract_embeddings_shape():
    tokenizer = load_tokenizer()
    tokens = tokenize_prompt("Test sentence", tokenizer)
    embeddings = extract_embeddings(tokens["input_ids"])
    assert embeddings.ndim == 2
    assert embeddings.shape[1] in (2048, 4096)
