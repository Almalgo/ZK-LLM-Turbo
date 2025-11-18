import torch
from transformers import AutoModel

def extract_embeddings(input_ids, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        outputs = model(input_ids)
        # Extract last hidden state as embeddings (batch_size, seq_len, hidden_size)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    return embeddings
