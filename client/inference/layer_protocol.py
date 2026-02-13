"""Encrypted layer protocol: orchestrates 4 round-trips per decoder layer.

Round 1: Q/K/V Projections
  Client: RMSNorm(X) → encrypt → send
  Server: Enc(Q)=Enc(X)@W_q, Enc(K)=Enc(X)@W_k, Enc(V)=Enc(X)@W_v

Round 2: O Projection
  Client: decrypt Q,K,V → RoPE → attention → encrypt
  Server: Enc(o_out) = Enc(attn) @ W_o

Round 3: FFN Gate + Up Projections
  Client: decrypt → residual + RMSNorm → encrypt
  Server: Enc(gate)=Enc(X)@W_gate, Enc(up)=Enc(X)@W_up (split output)

Round 4: FFN Down Projection
  Client: decrypt → SiLU(gate)*up → encrypt (split input)
  Server: Enc(down) = sum(Enc(part_i) @ W_down_i)
  Client: decrypt → residual → next layer input

All tokens in a step are batched into a single HTTP request per round.
KV cache is maintained across generation steps for incremental inference.
"""

import base64
import numpy as np
import requests
import tenseal as ts
from common.logging_utils import get_logger
from client.inference.nonlinear_ops import (
    rms_norm,
    silu,
    compute_attention_cached,
    apply_rotary_embeddings_at_positions,
)

logger = get_logger("client.protocol")

SLOT_COUNT = 4096  # poly_modulus_degree // 2


class EncryptedLayerProtocol:
    def __init__(
        self,
        context: ts.Context,
        session_id: str,
        server_url: str,
        layer_endpoint: str,
        auth_token: str,
        model_config,
    ):
        self.context = context
        self.session_id = session_id
        self.base_url = server_url
        self.layer_url = server_url + layer_endpoint
        self.auth_token = auth_token
        self.config = model_config
        self._kv_cache = {}  # layer_idx -> {"k": ndarray, "v": ndarray}

    def _encrypt_vector(self, vec: np.ndarray) -> ts.CKKSVector:
        return ts.ckks_vector(self.context, vec.tolist())

    def _decrypt_vector(self, enc_vec: ts.CKKSVector, expected_len: int) -> np.ndarray:
        dec = enc_vec.decrypt()
        return np.array(dec[:expected_len], dtype=np.float32)

    def _serialize_vectors(self, vectors: list[ts.CKKSVector]) -> list[str]:
        return [base64.b64encode(v.serialize()).decode("utf-8") for v in vectors]

    def _deserialize_vectors(self, vectors_b64: list[str]) -> list[ts.CKKSVector]:
        result = []
        for b64 in vectors_b64:
            raw = base64.b64decode(b64)
            vec = ts.ckks_vector_from(self.context, raw)
            result.append(vec)
        return result

    def _send_request(
        self, layer_idx: int, operation: str,
        enc_vectors: list[ts.CKKSVector],
        chunk_sizes: list[int] | None = None,
    ) -> list[ts.CKKSVector]:
        """Send encrypted vectors to server, get encrypted results back."""
        payload = {
            "session_id": self.session_id,
            "layer_idx": layer_idx,
            "operation": operation,
            "encrypted_vectors_b64": self._serialize_vectors(enc_vectors),
        }
        if chunk_sizes is not None:
            payload["chunk_sizes"] = chunk_sizes

        response = requests.post(
            self.layer_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        logger.info(
            f"Round {operation} complete",
            extra={"extra": {
                "layer": layer_idx, "op": operation,
                "server_ms": data.get("elapsed_ms"),
            }},
        )
        return self._deserialize_vectors(data["encrypted_results_b64"])

    def process_layer(
        self, hidden_states: np.ndarray, layer_idx: int,
        input_layernorm_weight: np.ndarray,
        post_attn_layernorm_weight: np.ndarray,
        eps: float,
        position_offset: int = 0,
    ) -> np.ndarray:
        """Process one decoder layer through 4 encrypted round-trips.

        All tokens are batched into single HTTP requests per round.
        KV cache is maintained across calls for incremental generation.

        hidden_states: (seq_len, hidden_dim) in plaintext
        position_offset: starting position index for RoPE
        Returns: (seq_len, hidden_dim) updated hidden states
        """
        seq_len = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = hidden_dim // num_heads
        intermediate_size = self.config.intermediate_size  # 5632

        # === Round 1: Q/K/V Projections (batched) ===
        residual = hidden_states.copy()
        normed = rms_norm(hidden_states, input_layernorm_weight, eps)

        enc_normed = [self._encrypt_vector(normed[t]) for t in range(seq_len)]
        results = self._send_request(layer_idx, "qkv", enc_normed)
        # Server returns [q0, k0, v0, q1, k1, v1, ...] = 3 per token

        all_q = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        all_k = np.zeros((seq_len, num_kv_heads * head_dim), dtype=np.float32)
        all_v = np.zeros((seq_len, num_kv_heads * head_dim), dtype=np.float32)
        for t in range(seq_len):
            all_q[t] = self._decrypt_vector(results[t * 3], hidden_dim)
            all_k[t] = self._decrypt_vector(results[t * 3 + 1], num_kv_heads * head_dim)
            all_v[t] = self._decrypt_vector(results[t * 3 + 2], num_kv_heads * head_dim)

        # === Client: RoPE + Attention ===
        positions = np.arange(position_offset, position_offset + seq_len)
        all_q, all_k = apply_rotary_embeddings_at_positions(
            all_q, all_k, positions,
            head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
        )

        # Update KV cache
        cache = self._kv_cache.get(layer_idx)
        if cache is not None:
            full_k = np.concatenate([cache["k"], all_k], axis=0)
            full_v = np.concatenate([cache["v"], all_v], axis=0)
        else:
            full_k = all_k
            full_v = all_v
        self._kv_cache[layer_idx] = {"k": full_k, "v": full_v}

        attn_output = compute_attention_cached(
            all_q, full_k, full_v,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        )

        # === Round 2: O Projection (batched) ===
        enc_attn = [self._encrypt_vector(attn_output[t]) for t in range(seq_len)]
        results = self._send_request(layer_idx, "o_proj", enc_attn)

        all_o = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        for t in range(seq_len):
            all_o[t] = self._decrypt_vector(results[t], hidden_dim)

        hidden_states = residual + all_o

        # === Round 3: FFN Gate + Up (batched) ===
        residual = hidden_states.copy()
        normed = rms_norm(hidden_states, post_attn_layernorm_weight, eps)

        chunk_sizes = []
        remaining = intermediate_size
        while remaining > 0:
            chunk = min(remaining, SLOT_COUNT)
            chunk_sizes.append(chunk)
            remaining -= chunk
        num_chunks = len(chunk_sizes)

        enc_normed = [self._encrypt_vector(normed[t]) for t in range(seq_len)]
        results = self._send_request(layer_idx, "ffn_gate_up", enc_normed)
        # Server returns per-token: [gate_c0, gate_c1, up_c0, up_c1]
        results_per_token = num_chunks * 2

        all_gate_chunks = [np.zeros((seq_len, cs), dtype=np.float32) for cs in chunk_sizes]
        all_up_chunks = [np.zeros((seq_len, cs), dtype=np.float32) for cs in chunk_sizes]
        for t in range(seq_len):
            base_idx = t * results_per_token
            for i in range(num_chunks):
                all_gate_chunks[i][t] = self._decrypt_vector(results[base_idx + i], chunk_sizes[i])
                all_up_chunks[i][t] = self._decrypt_vector(results[base_idx + num_chunks + i], chunk_sizes[i])

        all_gate = np.concatenate(all_gate_chunks, axis=-1)
        all_up = np.concatenate(all_up_chunks, axis=-1)
        ffn_hidden = silu(all_gate) * all_up

        # === Round 4: FFN Down (batched) ===
        enc_all = []
        for t in range(seq_len):
            offset = 0
            for cs in chunk_sizes:
                chunk_data = ffn_hidden[t, offset : offset + cs]
                enc_all.append(self._encrypt_vector(chunk_data))
                offset += cs
        results = self._send_request(layer_idx, "ffn_down", enc_all, chunk_sizes=chunk_sizes)

        all_down = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        for t in range(seq_len):
            all_down[t] = self._decrypt_vector(results[t], hidden_dim)

        hidden_states = residual + all_down
        return hidden_states
