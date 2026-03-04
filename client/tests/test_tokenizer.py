import pytest

from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt


@pytest.mark.slow
def test_tokenize_prompt():
    try:
        tokenizer = load_tokenizer()
    except OSError as exc:
        pytest.skip(f"Tokenizer unavailable in this environment: {exc}")
    result = tokenize_prompt("Hello world", tokenizer)
    assert "input_ids" in result
    assert len(result["input_ids"][0]) > 0
