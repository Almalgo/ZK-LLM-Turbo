import torch
from transformers import AutoModelForCausalLM

def extract_embeddings(input_ids, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Extract token embeddings using the model's embedding layer."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = model.model.embed_tokens

    with torch.no_grad():
        embeddings = embedding_layer(input_ids)
    
    return embeddings.squeeze(0).cpu().numpy()  # shape: (seq_len, hidden_dim)
