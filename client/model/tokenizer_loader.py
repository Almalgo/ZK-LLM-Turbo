from transformers import AutoTokenizer

def load_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load and return the TinyLlama tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_prompt(prompt: str, tokenizer):
    """Tokenize user input prompt."""
    return tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
