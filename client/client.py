"""ZK-LLM-Turbo Client: Privacy-preserving split inference with TinyLlama.

Orchestrates encrypted and plaintext layer processing:
  - Layers 0..N-1: encrypted via server (HE linear ops)
  - Layers N..21: plaintext via local PyTorch
  - Final: RMSNorm + lm_head → greedy token selection

Supports KV caching for fast incremental generation after the initial pass.
"""

import sys
import argparse
import logging
import time
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

from common.logging_utils import get_logger
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
    print("Loading model...")
    tokenizer = load_tokenizer()
    components = get_model_components()
    model_config = components["config"]
    total_layers = model_config.num_hidden_layers  # 22 for TinyLlama

    num_encrypted_layers = min(num_encrypted_layers, total_layers)

    # Create CKKS context
    context = create_ckks_context()

    # Setup server session
    print("Establishing encrypted session...")
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
    position_offset = 0
    plaintext_cache = None

    print(f'Generating {num_tokens} tokens for: "{prompt}"')

    for step in range(num_tokens):
        step_start = time.perf_counter()

        if step == 0:
            # Initial step: process all tokens
            embeddings = extract_embeddings(input_ids)  # (seq_len, 2048)
        else:
            # Incremental: process only the new token
            embeddings = extract_embeddings(input_ids[:, -1:])  # (1, 2048)

        hidden_states = embeddings
        curr_seq_len = hidden_states.shape[0]

        # Process encrypted layers
        for layer_idx in range(num_encrypted_layers):
            layer = components["layers"][layer_idx]
            input_ln_w = layer.input_layernorm.weight.detach().numpy()
            post_attn_ln_w = layer.post_attention_layernorm.weight.detach().numpy()
            eps = model_config.rms_norm_eps
            hidden_states = protocol.process_layer(
                hidden_states, layer_idx, input_ln_w, post_attn_ln_w, eps,
                position_offset=position_offset,
            )

        # Process plaintext layers with KV cache
        device = next(components["layers"][0].parameters()).device
        hidden_t = torch.tensor(hidden_states, dtype=torch.float32, device=device).unsqueeze(0)
        position_ids = torch.arange(
            position_offset, position_offset + curr_seq_len, device=device
        ).unsqueeze(0)

        if plaintext_cache is None:
            from transformers import DynamicCache
            plaintext_cache = DynamicCache()
            # Pre-fill cache slots for encrypted layers so layer indices align
            num_kv_heads = model_config.num_key_value_heads
            head_dim = model_config.hidden_size // model_config.num_attention_heads
            for _ in range(num_encrypted_layers):
                plaintext_cache.key_cache.append(
                    torch.zeros(1, num_kv_heads, 0, head_dim, device=device)
                )
                plaintext_cache.value_cache.append(
                    torch.zeros(1, num_kv_heads, 0, head_dim, device=device)
                )

        with torch.no_grad():
            for layer_idx in range(num_encrypted_layers, total_layers):
                layer = components["layers"][layer_idx]
                output = layer(
                    hidden_t,
                    position_ids=position_ids,
                    past_key_value=plaintext_cache,
                    use_cache=True,
                )
                hidden_t = output[0]

        hidden_states = hidden_t.squeeze(0).numpy()
        position_offset += curr_seq_len

        # Final RMSNorm + lm_head → logits (only last token)
        final_norm_w = components["final_norm_weight"]
        eps = model_config.rms_norm_eps
        last_hidden = rms_norm(hidden_states[-1:], final_norm_w, eps)

        lm_head_w = components["lm_head_weight"]  # (vocab_size, hidden_dim)
        logits = last_hidden @ lm_head_w.T  # (1, vocab_size)

        # Greedy decode
        next_token_id = int(np.argmax(logits[0]))
        generated_tokens.append(next_token_id)

        next_token_text = tokenizer.decode([next_token_id])
        elapsed = time.perf_counter() - step_start
        print(f"  Token {step + 1}/{num_tokens}: {next_token_text!r} ({elapsed:.1f}s)")

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

    print(f"\nFull output: {full_text}")
    return full_text


def main():
    parser = argparse.ArgumentParser(description="ZK-LLM-Turbo Client: privacy-preserving split inference")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for inference (interactive if omitted)")
    parser.add_argument("--num-tokens", type=int, default=5, help="Number of tokens to generate (default: 5)")
    parser.add_argument("--num-encrypted-layers", type=int, default=1, help="Number of initial layers processed encrypted (default: 1)")
    parser.add_argument("--logs", action="store_true", help="Enable verbose JSON logging output")
    args = parser.parse_args()

    if not args.logs:
        logging.disable(logging.CRITICAL)

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter your prompt: ")

    generate(
        prompt=prompt,
        num_tokens=args.num_tokens,
        num_encrypted_layers=args.num_encrypted_layers,
    )


if __name__ == "__main__":
    main()
