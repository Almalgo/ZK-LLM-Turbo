import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM

_cached_model = None
_cached_model_name = None


def _resolve_model_dtype() -> torch.dtype:
    dtype_name = os.getenv("ZKLLM_CLIENT_MODEL_DTYPE", "float32").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_name, torch.float32)


def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load and cache the model. Returns the full CausalLM model."""
    global _cached_model, _cached_model_name
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model
    model_dtype = _resolve_model_dtype()
    _cached_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    _cached_model.eval()
    _cached_model_name = model_name
    return _cached_model


def extract_embeddings(input_ids, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Extract initial token embeddings using embed_tokens only (no forward pass)."""
    model = load_model(model_name)
    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids).squeeze(0).numpy()
    return embeddings


def get_model_components(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Return model components needed for client-side operations.

    Returns dict with:
        - model: the full CausalLM model
        - layers: list of decoder layers
        - final_norm_weight: final RMSNorm weight
        - final_norm_eps: RMSNorm epsilon
        - lm_head_weight: language model head weight matrix
        - config: model config (num_heads, hidden_size, etc.)
    """
    model = load_model(model_name)
    config = model.config
    return {
        "model": model,
        "layers": model.model.layers,
        "final_norm_weight": model.model.norm.weight.detach().numpy(),
        "final_norm_eps": config.rms_norm_eps,
        "lm_head_weight": model.lm_head.weight.detach().numpy(),
        "config": config,
    }
