import torch
import numpy as np
from transformers import AutoModelForCausalLM
from common.logging_utils import get_logger

logger = get_logger("server.weights")

_model = None
_layer_weight_cache: dict[int, dict] = {}
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model():
    """Load TinyLlama model at server startup. Caches globally."""
    global _model
    if _model is not None:
        return _model
    logger.info("Loading model", extra={"extra": {"model": MODEL_NAME}})
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    _model.eval()
    logger.info("Model loaded", extra={"extra": {"model": MODEL_NAME}})
    return _model


def get_layer_weights(layer_idx: int) -> dict[str, np.ndarray]:
    """Extract and transpose weight matrices for a given layer.

    Linear layers store weights as (out_features, in_features).
    For HE matmul we need (in_features, out_features) so we transpose.

    Returns dict with keys: q_proj, k_proj, v_proj, o_proj,
                            gate_proj, up_proj, down_proj,
                            input_layernorm, post_attention_layernorm,
                            input_layernorm_eps
    """
    if layer_idx in _layer_weight_cache:
        return _layer_weight_cache[layer_idx]

    model = load_model()
    layer = model.model.layers[layer_idx]

    weights = {
        # Attention projections: transposed to (in, out) for x @ W
        "q_proj": layer.self_attn.q_proj.weight.detach().float().numpy().T,
        "k_proj": layer.self_attn.k_proj.weight.detach().float().numpy().T,
        "v_proj": layer.self_attn.v_proj.weight.detach().float().numpy().T,
        "o_proj": layer.self_attn.o_proj.weight.detach().float().numpy().T,
        # FFN projections: transposed to (in, out) for x @ W
        "gate_proj": layer.mlp.gate_proj.weight.detach().float().numpy().T,
        "up_proj": layer.mlp.up_proj.weight.detach().float().numpy().T,
        "down_proj": layer.mlp.down_proj.weight.detach().float().numpy().T,
        # LayerNorm weights (not transposed - 1D vectors)
        "input_layernorm": layer.input_layernorm.weight.detach().float().numpy(),
        "post_attention_layernorm": layer.post_attention_layernorm.weight.detach().float().numpy(),
    }

    _layer_weight_cache[layer_idx] = weights
    logger.info("Layer weights cached", extra={"extra": {"layer_idx": layer_idx}})
    return weights
