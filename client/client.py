"""ZK-LLM-Turbo Client: Privacy-preserving split inference with TinyLlama.

Orchestrates encrypted and plaintext layer processing:
  - Layers 0..N-1: encrypted via server (HE linear ops)
  - Layers N..21: plaintext via local PyTorch
  - Final: RMSNorm + lm_head → greedy token selection
"""

import sys
from pathlib import Path
import base64
import uuid
import requests
import yaml
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.logging_utils import get_logger, timed_execution
from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt
from client.model.embedding_extractor import extract_embeddings, get_model_components
from client.encryption.ckks_context import create_ckks_context, serialize_public_context
from client.inference.layer_protocol import EncryptedLayerProtocol
from client.inference.nonlinear_ops import rms_norm

logger = get_logger("client")


def load_config():
    """Load client and server configuration."""
    client_cfg = yaml.safe_load(Path("client/config/client_config.yaml").read_text())
    server_cfg = yaml.safe_load(Path("client/config/endpoints.yaml").read_text())["server"]
    return client_cfg, server_cfg


def setup_session(context, server_cfg):
    """Send public context to server, get session ID."""
    public_bytes = serialize_public_context(context)
    public_b64 = base64.b64encode(public_bytes).decode("utf-8")

    url = server_cfg["base_url"] + server_cfg["session_endpoint"]
    response = requests.post(
        url,
        json={"public_context_b64": public_b64},
        headers={
            "Authorization": f"Bearer {server_cfg['auth_token']}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    session_id = response.json()["session_id"]
    logger.info("Session established", extra={"extra": {"session_id": session_id}})
    return session_id


def process_layer_plaintext(hidden_states_np, layer, config):
    """Process a single decoder layer using PyTorch (plaintext)."""
    device = next(layer.parameters()).device
    hidden_states = torch.tensor(hidden_states_np, dtype=torch.float32, device=device).unsqueeze(0)
    seq_len = hidden_states.shape[1]

    # Create position_ids and causal mask
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        output = layer(hidden_states, position_ids=position_ids)
        # LlamaDecoderLayer returns (hidden_states, ...) tuple
        result = output[0].squeeze(0).numpy()

    return result


def generate(prompt: str, num_tokens: int = 5, num_encrypted_layers: int = 1):
    """Generate tokens using split encrypted/plaintext inference.

    Args:
        prompt: input text
        num_tokens: number of tokens to generate
        num_encrypted_layers: how many initial layers to process encrypted (default 1)
    """
    cid = str(uuid.uuid4())
    logger.info("Starting generation", extra={"extra": {
        "cid": cid, "prompt": prompt[:50],
        "num_tokens": num_tokens, "encrypted_layers": num_encrypted_layers,
    }})

    # Load configs
    client_cfg, server_cfg = load_config()

    # Load tokenizer and model
    tokenizer = load_tokenizer()
    components = get_model_components()
    model_config = components["config"]
    total_layers = model_config.num_hidden_layers  # 22 for TinyLlama

    num_encrypted_layers = min(num_encrypted_layers, total_layers)

    # Create CKKS context
    context = create_ckks_context()

    # Setup server session
    session_id = setup_session(context, server_cfg)

    # Create layer protocol for encrypted rounds
    protocol = EncryptedLayerProtocol(
        context=context,
        session_id=session_id,
        server_url=server_cfg["base_url"],
        layer_endpoint=server_cfg["layer_endpoint"],
        auth_token=server_cfg["auth_token"],
        model_config=model_config,
    )

    # Tokenize
    tokens = tokenize_prompt(prompt, tokenizer)
    input_ids = tokens["input_ids"]

    generated_tokens = []

    for step in range(num_tokens):
        logger.info(f"Generation step {step}", extra={"extra": {"cid": cid}})

        # Extract initial embeddings (embed_tokens only)
        with timed_execution(logger, "Embedding extraction"):
            embeddings = extract_embeddings(input_ids)  # (seq_len, 2048)

        hidden_states = embeddings

        # Process encrypted layers (0 to num_encrypted_layers-1)
        for layer_idx in range(num_encrypted_layers):
            with timed_execution(logger, f"Encrypted layer {layer_idx}"):
                layer = components["layers"][layer_idx]
                input_ln_w = layer.input_layernorm.weight.detach().numpy()
                post_attn_ln_w = layer.post_attention_layernorm.weight.detach().numpy()
                eps = model_config.rms_norm_eps
                hidden_states = protocol.process_layer(
                    hidden_states, layer_idx, input_ln_w, post_attn_ln_w, eps
                )

        # Process plaintext layers (num_encrypted_layers to total_layers-1)
        for layer_idx in range(num_encrypted_layers, total_layers):
            with timed_execution(logger, f"Plaintext layer {layer_idx}"):
                layer = components["layers"][layer_idx]
                hidden_states = process_layer_plaintext(hidden_states, layer, model_config)

        # Final RMSNorm + lm_head → logits
        final_norm_w = components["final_norm_weight"]
        eps = model_config.rms_norm_eps
        hidden_states = rms_norm(hidden_states, final_norm_w, eps)

        # lm_head: (vocab_size, hidden_dim) — standard matmul
        lm_head_w = components["lm_head_weight"]  # (vocab_size, hidden_dim)
        logits = hidden_states[-1:] @ lm_head_w.T  # (1, vocab_size)

        # Greedy decode
        next_token_id = int(np.argmax(logits[0]))
        generated_tokens.append(next_token_id)

        next_token_text = tokenizer.decode([next_token_id])
        logger.info(f"Generated token: {next_token_text}", extra={"extra": {
            "cid": cid, "token_id": next_token_id, "step": step,
        }})

        # Append token for next step
        new_token = torch.tensor([[next_token_id]], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, new_token], dim=1)

    # Decode full output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + output_text

    logger.info("Generation complete", extra={"extra": {
        "cid": cid, "generated": output_text, "full": full_text,
    }})

    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output_text}")
    print(f"Full: {full_text}")
    return full_text


if __name__ == "__main__":
    generate(
        prompt="The capital of France is",
        num_tokens=5,
        num_encrypted_layers=1,
    )
