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
import asyncio
import time
import numpy as np
import requests
import msgpack
import zstandard as zstd
import tenseal as ts
from websockets.sync.client import connect
from common.constants import SLOT_COUNT
from common.logging_utils import get_logger
from client.inference.nonlinear_ops import (
    rms_norm,
    silu,
    poly_silu,
    compute_attention_cached,
    apply_rotary_embeddings_at_positions,
)

logger = get_logger("client.protocol")

_zstd_compressor = zstd.ZstdCompressor(level=3)
_zstd_decompressor = zstd.ZstdDecompressor()


class EncryptedLayerProtocol:
    def __init__(
        self,
        context: ts.Context,
        session_id: str,
        server_url: str,
        layer_endpoint: str,
        auth_token: str,
        model_config,
        websocket_layer_endpoint: str | None = None,
        use_merged_ffn: bool = False,
        use_poly_silu: bool = False,
        use_websocket: bool = False,
    ):
        self.context = context
        self.session_id = session_id
        self.base_url = server_url
        self.layer_url = server_url + layer_endpoint
        self.auth_token = auth_token
        self.config = model_config
        self.websocket_url = self._build_websocket_url(
            server_url,
            websocket_layer_endpoint or f"{layer_endpoint}/ws",
        )
        self.use_merged_ffn = use_merged_ffn
        self.use_poly_silu = use_poly_silu
        self.use_websocket = use_websocket
        self._kv_cache = {}  # layer_idx -> {"k": ndarray, "v": ndarray}
        self._round_metrics = []
        self._websocket = None

        # Reuse HTTP connection across all requests
        self._http_session = requests.Session()
        self._http_session.headers.update({
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        })

    @staticmethod
    def _build_websocket_url(server_url: str, websocket_layer_endpoint: str) -> str:
        if server_url.startswith("https://"):
            ws_base = "wss://" + server_url[len("https://") :]
        elif server_url.startswith("http://"):
            ws_base = "ws://" + server_url[len("http://") :]
        else:
            ws_base = server_url
        return ws_base.rstrip("/") + websocket_layer_endpoint

    def _ensure_websocket(self):
        if self._websocket is None:
            self._websocket = connect(
                self.websocket_url,
                additional_headers={"Authorization": f"Bearer {self.auth_token}"},
                compression=None,
                open_timeout=30,
                close_timeout=10,
                max_size=None,
            )
        return self._websocket

    def close(self) -> None:
        if self._websocket is not None:
            self._websocket.close()
            self._websocket = None
        self._http_session.close()

    def reset_round_metrics(self) -> None:
        """Clear captured per-round transport metrics."""
        self._round_metrics.clear()

    def get_round_metrics(self) -> list[dict]:
        """Return captured per-round transport metrics."""
        return list(self._round_metrics)

    def _encrypt_vector(self, vec: np.ndarray) -> ts.CKKSVector:
        return ts.ckks_vector(self.context, vec.tolist())

    def _decrypt_vector(self, enc_vec: ts.CKKSVector, expected_len: int) -> np.ndarray:
        dec = enc_vec.decrypt()
        return np.array(dec[:expected_len], dtype=np.float32)

    def _pack_tokens(
        self,
        vectors: list[np.ndarray],
        dim: int,
        max_pack: int = 2,
    ) -> list[tuple[ts.CKKSVector, int]]:
        """Pack multiple same-dimension vectors into one ciphertext when slots allow it."""
        if not vectors:
            return []

        pack_width = max(1, min(max_pack, SLOT_COUNT // dim))
        packed = []
        for start in range(0, len(vectors), pack_width):
            chunk = vectors[start : start + pack_width]
            pack_count = len(chunk)
            padded = np.zeros(pack_width * dim, dtype=np.float32)
            for idx, vec in enumerate(chunk):
                padded[idx * dim : (idx + 1) * dim] = vec[:dim]
            packed.append((self._encrypt_vector(padded), pack_count))
        return packed

    def _unpack_tokens(
        self,
        enc_vec: ts.CKKSVector,
        pack_count: int,
        dim: int,
    ) -> list[np.ndarray]:
        """Recover packed plaintext vectors from one decrypted ciphertext."""
        dec = np.array(enc_vec.decrypt(), dtype=np.float32)
        return [
            dec[idx * dim : (idx + 1) * dim].astype(np.float32)
            for idx in range(pack_count)
        ]

    def _serialize_vectors(self, vectors: list[ts.CKKSVector]) -> list[bytes]:
        """Serialize and compress encrypted vectors with zstd."""
        return [_zstd_compressor.compress(v.serialize()) for v in vectors]

    def _deserialize_vectors(self, vectors_compressed: list[bytes]) -> list[ts.CKKSVector]:
        """Decompress and deserialize encrypted vectors."""
        return [ts.ckks_vector_from(self.context, _zstd_decompressor.decompress(raw)) for raw in vectors_compressed]

    def _serialize_vectors_b64(self, vectors: list[ts.CKKSVector]) -> list[str]:
        """Legacy base64 serialization for JSON fallback."""
        return [base64.b64encode(v.serialize()).decode("utf-8") for v in vectors]

    def _deserialize_vectors_b64(self, vectors_b64: list[str]) -> list[ts.CKKSVector]:
        """Legacy base64 deserialization for JSON fallback."""
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
        pack_counts: list[int] | None = None,
    ) -> list[ts.CKKSVector]:
        """Send encrypted vectors to server via msgpack binary transport.

        Falls back to JSON+base64 if the binary endpoint is unavailable.
        Includes detailed timing instrumentation for performance analysis.
        """
        # Time serialization + compression
        t0 = time.perf_counter()
        serialized = self._serialize_vectors(enc_vectors)
        serialize_ms = (time.perf_counter() - t0) * 1000

        payload = {
            "session_id": self.session_id,
            "layer_idx": layer_idx,
            "operation": operation,
            "encrypted_vectors": serialized,
        }
        if chunk_sizes is not None:
            payload["chunk_sizes"] = chunk_sizes
        if pack_counts is not None:
            payload["pack_counts"] = pack_counts

        # Time network roundtrip (pack + send + receive)
        t0 = time.perf_counter()
        body = msgpack.packb(payload, use_bin_type=True)
        payload_bytes = len(body)
        used_binary = True
        response_content = b""

        if self.use_websocket:
            try:
                ws = self._ensure_websocket()
                ws.send(body)
                response_content = ws.recv()
            except Exception:
                self.close()
                self.use_websocket = False
                logger.warning(
                    "WebSocket transport unavailable, falling back to HTTP",
                    extra={"extra": {"layer": layer_idx, "op": operation}},
                )

        if not response_content:
            try:
                response = self._http_session.post(
                    self.layer_url + "/binary",
                    data=body,
                    headers={"Content-Type": "application/msgpack"},
                    timeout=300,
                )
                response.raise_for_status()
                response_content = response.content
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
                used_binary = False
                vectors_b64 = [base64.b64encode(raw).decode("utf-8") for raw in
                               [_zstd_decompressor.decompress(s) for s in serialized]]
                fallback_payload = {
                    "session_id": self.session_id,
                    "layer_idx": layer_idx,
                    "operation": operation,
                    "encrypted_vectors_b64": vectors_b64,
                }
                if chunk_sizes is not None:
                    fallback_payload["chunk_sizes"] = chunk_sizes
                if pack_counts is not None:
                    fallback_payload["pack_counts"] = pack_counts
                response = self._http_session.post(
                    self.layer_url,
                    json=fallback_payload,
                    timeout=300,
                )
                response.raise_for_status()
                response_content = response.content
                payload_bytes = len(response.request.body) if response.request.body else 0

        roundtrip_ms = (time.perf_counter() - t0) * 1000

        # Time deserialization + decompression
        t0 = time.perf_counter()
        if used_binary:
            data = msgpack.unpackb(response_content, raw=False)
            result = self._deserialize_vectors(data["encrypted_results"])
        else:
            data = response.json()
            result = self._deserialize_vectors_b64(data["encrypted_results_b64"])
        deserialize_ms = (time.perf_counter() - t0) * 1000

        server_ms = data.get("elapsed_ms", 0)
        network_ms = max(0, roundtrip_ms - server_ms)

        metrics = {
            "layer": layer_idx,
            "op": operation,
            "transport": "websocket" if self.use_websocket and used_binary else ("binary" if used_binary else "json"),
            "serialize_ms": round(serialize_ms, 1),
            "server_ms": round(server_ms, 1),
            "network_ms": round(network_ms, 1),
            "deserialize_ms": round(deserialize_ms, 1),
            "roundtrip_ms": round(roundtrip_ms, 1),
            "payload_kb": round(payload_bytes / 1024, 1),
            "response_kb": round(len(response_content) / 1024, 1),
        }
        self._round_metrics.append(metrics)

        logger.info(
            f"Round {operation} complete",
            extra={"extra": metrics},
        )
        return result

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
        if self.use_merged_ffn:
            merged_results = self._send_request(
                layer_idx,
                "ffn_merged",
                enc_normed,
                chunk_sizes=chunk_sizes,
            )
            all_down = np.zeros((seq_len, hidden_dim), dtype=np.float32)
            for t in range(seq_len):
                all_down[t] = self._decrypt_vector(merged_results[t], hidden_dim)
        else:
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
            activation_fn = poly_silu if self.use_poly_silu else silu
            ffn_hidden = activation_fn(all_gate) * all_up

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

    async def process_layer_async(
        self,
        hidden_states: np.ndarray,
        layer_idx: int,
        input_layernorm_weight: np.ndarray,
        post_attn_layernorm_weight: np.ndarray,
        eps: float,
        position_offset: int = 0,
    ) -> np.ndarray:
        return self.process_layer(
            hidden_states,
            layer_idx,
            input_layernorm_weight,
            post_attn_layernorm_weight,
            eps,
            position_offset,
        )
