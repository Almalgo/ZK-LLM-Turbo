"""ZK-LLM-Turbo Client: Privacy-preserving split inference with TinyLlama.

Orchestrates encrypted and plaintext layer processing:
  - Layers 0..N-1: encrypted via server (HE linear ops)
  - Layers N..21: plaintext via local PyTorch
  - Final: RMSNorm + lm_head → greedy token selection

Supports KV caching for fast incremental generation after the initial pass.
"""

import sys
import argparse
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
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
from client.inference.pipeline import (
    get_encrypted_layer_params,
    run_encrypted_layers,
    run_encrypted_layers_async,
)
from client.inference.nonlinear_ops import rms_norm

logger = get_logger("client")

# Module-level reusable HTTP session
_http_session = requests.Session()


@dataclass(frozen=True)
class GenerationResult:
    full_text: str
    generated_text: str
    generated_token_ids: list[int]
    tokens_generated: int
    stats: dict[str, float]


def load_config():
    """Load client and server configuration."""
    client_cfg = yaml.safe_load(Path("client/config/client_config.yaml").read_text())
    server_cfg = yaml.safe_load(Path("client/config/endpoints.yaml").read_text())["server"]

    env_overrides = {
        "base_url": os.getenv("ZKLLM_SERVER_BASE_URL"),
        "infer_endpoint": os.getenv("ZKLLM_SERVER_INFER_ENDPOINT"),
        "session_endpoint": os.getenv("ZKLLM_SERVER_SESSION_ENDPOINT"),
        "layer_endpoint": os.getenv("ZKLLM_SERVER_LAYER_ENDPOINT"),
        "layer_ws_endpoint": os.getenv("ZKLLM_SERVER_LAYER_WS_ENDPOINT"),
        "auth_token": os.getenv("ZKLLM_SERVER_AUTH_TOKEN"),
    }
    for key, value in env_overrides.items():
        if value:
            server_cfg[key] = value

    request_timeout_seconds = os.getenv("ZKLLM_REQUEST_TIMEOUT_SECONDS")
    if request_timeout_seconds:
        client_cfg.setdefault("inference", {})["request_timeout_seconds"] = float(
            request_timeout_seconds
        )

    return client_cfg, server_cfg


def setup_session(context, server_cfg):
    """Send public context to server, get session ID."""
    public_bytes = serialize_public_context(context)
    public_b64 = base64.b64encode(public_bytes).decode("utf-8")

    url = server_cfg["base_url"] + server_cfg["session_endpoint"]
    # Use per-request headers instead of session-level to avoid leaking
    # auth tokens if setup_session is called with different server configs.
    headers = {
        "Authorization": f"Bearer {server_cfg['auth_token']}",
        "Content-Type": "application/json",
    }
    max_attempts = int(server_cfg.get("session_setup_max_attempts", 3))
    retry_delay_seconds = float(server_cfg.get("session_setup_retry_delay_seconds", 2.0))

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = _http_session.post(
                url,
                json={"public_context_b64": public_b64},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            session_id = response.json()["session_id"]
            break
        except (requests.exceptions.RequestException, KeyError) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            jitter = random.uniform(0.0, 0.25)
            sleep_s = retry_delay_seconds * attempt + jitter
            logger.warning(
                "Session setup failed, retrying",
                extra={
                    "extra": {
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "retry_delay_seconds": round(sleep_s, 2),
                        "error": str(exc),
                    }
                },
            )
            time.sleep(sleep_s)

    if last_error is not None and "session_id" not in locals():
        raise last_error

    logger.info("Session established", extra={"extra": {"session_id": session_id}})
    return session_id


def _print_stats(stats):
    """Print a timing breakdown table."""
    rows = [
        ("Model loading", stats["model_loading"]),
        ("Session setup", stats["session_setup"]),
        ("Embedding extraction", stats["embedding"]),
        ("Encrypted layers", stats["encrypted"]),
        ("Plaintext layers", stats["plaintext"]),
        ("Token decode", stats["decode"]),
    ]
    total = stats["total"]

    print("\n+-------------------------+----------+--------+")
    print("| Phase                   | Time (s) |      % |")
    print("+-------------------------+----------+--------+")
    for label, t in rows:
        pct = (t / total * 100) if total > 0 else 0
        print(f"| {label:<23s} | {t:>8.2f} | {pct:>5.1f}% |")
    print("+-------------------------+----------+--------+")
    print(f"| {'Total':<23s} | {total:>8.2f} |        |")
    print("+-------------------------+----------+--------+")


def generate(prompt: str, num_tokens: int = 5, num_encrypted_layers: int = 1,
             show_stats: bool = False, return_stats: bool = False,
             quiet: bool = False,
             use_websocket_override: bool | None = None,
             use_async_pipeline_override: bool | None = None,
             use_merged_ffn_override: bool | None = None,
             use_poly_silu_override: bool | None = None):
    """Generate tokens using split encrypted/plaintext inference.

    Args:
        prompt: input text
        num_tokens: number of tokens to generate
        num_encrypted_layers: how many initial layers to process encrypted (default 1)
        show_stats: print timing breakdown table after generation
    """
    stats = {
        "model_loading": 0.0,
        "session_setup": 0.0,
        "embedding": 0.0,
        "encrypted": 0.0,
        "plaintext": 0.0,
        "decode": 0.0,
        "total": 0.0,
    }
    total_start = time.perf_counter()
    pipeline_enabled = False

    cid = str(uuid.uuid4())
    logger.info("Starting generation", extra={"extra": {
        "cid": cid, "prompt": prompt[:50],
        "num_tokens": num_tokens, "encrypted_layers": num_encrypted_layers,
    }})

    # Load configs
    client_cfg, server_cfg = load_config()

    # Load tokenizer and model
    if not quiet:
        print("Loading model...")
    t0 = time.perf_counter()
    tokenizer = load_tokenizer()
    components = get_model_components()
    model_config = components["config"]
    total_layers = model_config.num_hidden_layers  # 22 for TinyLlama
    num_encrypted_layers = min(num_encrypted_layers, total_layers)
    context = create_ckks_context()
    encrypted_layer_params = get_encrypted_layer_params(
        components,
        model_config,
        num_encrypted_layers,
    )
    stats["model_loading"] = time.perf_counter() - t0

    # Setup server session
    if not quiet:
        print("Establishing encrypted session...")
    t0 = time.perf_counter()
    session_id = setup_session(context, server_cfg)
    stats["session_setup"] = time.perf_counter() - t0

    # Create layer protocol for encrypted rounds
    use_websocket = client_cfg.get("inference", {}).get("use_websocket", False)
    if use_websocket_override is not None:
        use_websocket = use_websocket_override

    use_merged_ffn = client_cfg.get("inference", {}).get("use_merged_ffn", False)
    if use_merged_ffn_override is not None:
        use_merged_ffn = use_merged_ffn_override

    use_poly_silu = client_cfg.get("inference", {}).get("use_poly_silu", False)
    if use_poly_silu_override is not None:
        use_poly_silu = use_poly_silu_override

    protocol = EncryptedLayerProtocol(
        context=context,
        session_id=session_id,
        server_url=server_cfg["base_url"],
        layer_endpoint=server_cfg["layer_endpoint"],
        auth_token=server_cfg["auth_token"],
        model_config=model_config,
        websocket_layer_endpoint=server_cfg.get("layer_ws_endpoint"),
        use_merged_ffn=use_merged_ffn,
        use_poly_silu=use_poly_silu,
        use_websocket=use_websocket,
        request_timeout_seconds=client_cfg.get("inference", {}).get(
            "request_timeout_seconds", 300
        ),
    )
    pipeline_enabled = client_cfg.get("inference", {}).get("use_async_pipeline", False)
    if use_async_pipeline_override is not None:
        pipeline_enabled = use_async_pipeline_override

    # Tokenize
    tokens = tokenize_prompt(prompt, tokenizer)
    input_ids = tokens["input_ids"]

    generated_tokens = []
    position_offset = 0
    plaintext_cache = None

    if not quiet:
        print(f'Generating {num_tokens} tokens for: "{prompt}"')

    try:
        for step in range(num_tokens):
            step_start = time.perf_counter()

            # --- Embedding extraction ---
            t0 = time.perf_counter()
            if step == 0:
                embeddings = extract_embeddings(input_ids)
            else:
                embeddings = extract_embeddings(input_ids[:, -1:])
            stats["embedding"] += time.perf_counter() - t0

            hidden_states = embeddings
            curr_seq_len = hidden_states.shape[0]

            # --- Encrypted layers ---
            t0 = time.perf_counter()
            if pipeline_enabled:
                hidden_states = asyncio.run(
                    run_encrypted_layers_async(
                        hidden_states,
                        protocol,
                        encrypted_layer_params,
                        position_offset,
                    )
                )
            else:
                hidden_states = run_encrypted_layers(
                    hidden_states,
                    protocol,
                    encrypted_layer_params,
                    position_offset,
                )
            stats["encrypted"] += time.perf_counter() - t0

            # --- Plaintext layers ---
            t0 = time.perf_counter()
            device = next(components["layers"][0].parameters()).device
            hidden_t = torch.tensor(hidden_states, dtype=torch.float32, device=device).unsqueeze(0)
            position_ids = torch.arange(
                position_offset, position_offset + curr_seq_len, device=device
            ).unsqueeze(0)

            if plaintext_cache is None:
                from transformers import DynamicCache
                plaintext_cache = DynamicCache()
                if hasattr(plaintext_cache, "key_cache") and hasattr(plaintext_cache, "value_cache"):
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
                position_embeddings = components["model"].model.rotary_emb(hidden_t, position_ids)
                for layer_idx in range(num_encrypted_layers, total_layers):
                    layer = components["layers"][layer_idx]
                    output = layer(
                        hidden_t,
                        position_ids=position_ids,
                        past_key_value=plaintext_cache,
                        use_cache=True,
                        position_embeddings=position_embeddings,
                    )
                    hidden_t = output[0]

            hidden_states = hidden_t.squeeze(0).numpy()
            position_offset += curr_seq_len
            stats["plaintext"] += time.perf_counter() - t0

            # --- Token decode (final norm + lm_head + argmax) ---
            t0 = time.perf_counter()
            final_norm_w = components["final_norm_weight"]
            eps = model_config.rms_norm_eps
            last_hidden = rms_norm(hidden_states[-1:], final_norm_w, eps)

            lm_head_w = components["lm_head_weight"]
            logits = last_hidden @ lm_head_w.T

            next_token_id = int(np.argmax(logits[0]))
            generated_tokens.append(next_token_id)
            stats["decode"] += time.perf_counter() - t0

            next_token_text = tokenizer.decode([next_token_id])
            elapsed = time.perf_counter() - step_start
            if not quiet:
                print(f"  Token {step + 1}/{num_tokens}: {next_token_text!r} ({elapsed:.1f}s)")

            logger.info(f"Generated token: {next_token_text}", extra={"extra": {
                "cid": cid, "token_id": next_token_id, "step": step,
            }})

            new_token = torch.tensor([[next_token_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, new_token], dim=1)
    finally:
        protocol.close()

    # Decode full output
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + output_text
    stats["total"] = time.perf_counter() - total_start

    logger.info("Generation complete", extra={"extra": {
        "cid": cid, "generated": output_text, "full": full_text,
    }})

    if not quiet:
        print(f"\nFull output: {full_text}")

    if show_stats and not quiet:
        _print_stats(stats)

    if return_stats:
        return GenerationResult(
            full_text=full_text,
            generated_text=output_text,
            generated_token_ids=generated_tokens,
            tokens_generated=len(generated_tokens),
            stats=stats,
        )

    return full_text


def main():
    parser = argparse.ArgumentParser(description="ZK-LLM-Turbo Client: privacy-preserving split inference")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for inference (interactive if omitted)")
    parser.add_argument("--num-tokens", type=int, default=5, help="Number of tokens to generate (default: 5)")
    parser.add_argument("--num-encrypted-layers", type=int, default=1, help="Number of initial layers processed encrypted (default: 1)")
    parser.add_argument("--logs", action="store_true", help="Enable verbose JSON logging output")
    parser.add_argument("--stats", action="store_true", help="Print timing breakdown table after generation")
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
        show_stats=args.stats,
    )


if __name__ == "__main__":
    main()
