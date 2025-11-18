from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt

def test_tokenize_prompt():
    tokenizer = load_tokenizer()
    result = tokenize_prompt("Hello world", tokenizer)
    assert "input_ids" in result
    assert len(result["input_ids"][0]) > 0
