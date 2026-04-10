import asyncio
from types import SimpleNamespace

import msgpack
import numpy as np

from client.inference import layer_protocol as layer_protocol_module
from client.inference.layer_protocol import EncryptedLayerProtocol


def test_build_websocket_url_from_http():
    assert (
        EncryptedLayerProtocol._build_websocket_url(
            "http://localhost:8000",
            "/api/layer/ws",
        )
        == "ws://localhost:8000/api/layer/ws"
    )


def test_build_websocket_url_from_https():
    assert (
        EncryptedLayerProtocol._build_websocket_url(
            "https://example.com",
            "/api/layer/ws",
        )
        == "wss://example.com/api/layer/ws"
    )


def test_send_request_async_wraps_sync_request():
    protocol = object.__new__(EncryptedLayerProtocol)
    calls = []

    def fake_send_request(layer_idx, operation, enc_vectors, chunk_sizes=None, pack_counts=None):
        calls.append((layer_idx, operation, enc_vectors, chunk_sizes, pack_counts))
        return ["ok"]

    protocol._send_request = fake_send_request

    result = asyncio.run(
        protocol._send_request_async(3, "qkv", ["enc"], chunk_sizes=[4], pack_counts=[1])
    )

    assert result == ["ok"]
    assert calls == [(3, "qkv", ["enc"], [4], [1])]


def test_process_layer_async_uses_async_round_flow(monkeypatch):
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol.config = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=3,
    )
    protocol.use_merged_ffn = True
    protocol.use_poly_silu = False
    protocol._kv_cache = {}
    protocol.process_layer = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("sync path should not be used")
    )

    async def fake_encrypt_vectors_async(vectors):
        return [f"enc:{idx}" for idx, _ in enumerate(vectors)]

    calls = []

    async def fake_send_request_async(layer_idx, operation, enc_vectors, chunk_sizes=None, pack_counts=None):
        calls.append((operation, len(enc_vectors), chunk_sizes))
        if operation == "qkv":
            return ["q0", "k0", "v0"]
        if operation == "o_proj":
            return ["o0"]
        if operation == "ffn_merged":
            return ["d0"]
        raise AssertionError(f"unexpected operation {operation}")

    decrypted = {
        "q0": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "k0": np.array([5.0, 6.0], dtype=np.float32),
        "v0": np.array([7.0, 8.0], dtype=np.float32),
        "o0": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "d0": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }

    async def fake_decrypt_pairs_async(decrypt_pairs):
        return [decrypted[enc_vec] for enc_vec, _ in decrypt_pairs]

    protocol._encrypt_vectors_async = fake_encrypt_vectors_async
    protocol._send_request_async = fake_send_request_async
    protocol._decrypt_pairs_async = fake_decrypt_pairs_async

    monkeypatch.setattr(layer_protocol_module, "rms_norm", lambda x, weight, eps: x)
    monkeypatch.setattr(
        layer_protocol_module,
        "apply_rotary_embeddings_at_positions",
        lambda q, k, positions, head_dim, num_heads, num_kv_heads: (q, k),
    )
    monkeypatch.setattr(
        layer_protocol_module,
        "compute_attention_cached",
        lambda q, k, v, num_heads, num_kv_heads, head_dim: np.full((1, 4), 0.25, dtype=np.float32),
    )

    result = asyncio.run(
        protocol.process_layer_async(
            np.zeros((1, 4), dtype=np.float32),
            layer_idx=0,
            input_layernorm_weight=np.ones(4, dtype=np.float32),
            post_attn_layernorm_weight=np.ones(4, dtype=np.float32),
            eps=1e-5,
            position_offset=0,
        )
    )

    np.testing.assert_allclose(result, np.full((1, 4), 1.5, dtype=np.float32))
    assert calls == [
        ("qkv", 1, None),
        ("o_proj", 1, None),
        ("ffn_merged", 1, [3]),
    ]


class _FakeResponse:
    def __init__(self, content, json_data=None):
        self.content = content
        self._json_data = json_data
        self.request = SimpleNamespace(body=None)

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def close(self):
        return None


def test_send_request_uses_websocket_binary_transport():
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol.session_id = "session-1"
    protocol.use_websocket = True
    protocol._round_metrics = []
    protocol.layer_url = "http://server/api/layer"
    protocol._http_session = _FakeSession(response=None)
    protocol._websocket = None
    protocol._serialize_vectors = lambda vectors: [b"enc-a"]
    protocol._deserialize_vectors = lambda vectors: [f"decoded:{vectors[0]!r}"]

    sent_payloads = []

    class FakeWebSocket:
        def send(self, body):
            sent_payloads.append(msgpack.unpackb(body, raw=False))

        def recv(self):
            return msgpack.packb(
                {"encrypted_results": [b"server-a"], "elapsed_ms": 7.5},
                use_bin_type=True,
            )

    protocol._ensure_websocket = lambda: FakeWebSocket()

    result = protocol._send_request(2, "qkv", ["cipher"], chunk_sizes=[4], pack_counts=[1])

    assert result == ["decoded:b'server-a'"]
    assert sent_payloads == [
        {
            "session_id": "session-1",
            "layer_idx": 2,
            "operation": "qkv",
            "encrypted_vectors": [b"enc-a"],
            "chunk_sizes": [4],
            "pack_counts": [1],
        }
    ]
    assert protocol._http_session.calls == []
    assert protocol.get_round_metrics()[-1]["transport"] == "websocket"


def test_send_request_falls_back_to_http_after_websocket_failure():
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol.session_id = "session-2"
    protocol.use_websocket = True
    protocol._round_metrics = []
    protocol.layer_url = "http://server/api/layer"
    protocol._serialize_vectors = lambda vectors: [b"enc-b"]
    protocol._deserialize_vectors = lambda vectors: [f"decoded:{vectors[0]!r}"]
    protocol._websocket = object()
    protocol.close = lambda: setattr(protocol, "_websocket", None)

    response = _FakeResponse(
        msgpack.packb({"encrypted_results": [b"server-b"], "elapsed_ms": 3.0}, use_bin_type=True)
    )
    protocol._http_session = _FakeSession(response=response)

    class BrokenWebSocket:
        def send(self, body):
            raise RuntimeError("socket closed")

    protocol._ensure_websocket = lambda: BrokenWebSocket()

    result = protocol._send_request(1, "o_proj", ["cipher"])

    assert result == ["decoded:b'server-b'"]
    assert protocol.use_websocket is False
    assert protocol._http_session.calls[0][0] == "http://server/api/layer/binary"
    assert protocol.get_round_metrics()[-1]["transport"] == "binary"


def test_send_request_uses_configured_http_timeout():
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol.session_id = "session-3"
    protocol.use_websocket = False
    protocol.request_timeout_seconds = 123
    protocol._round_metrics = []
    protocol.layer_url = "http://server/api/layer"
    protocol._serialize_vectors = lambda vectors: [b"enc-c"]
    protocol._deserialize_vectors = lambda vectors: [f"decoded:{vectors[0]!r}"]
    protocol._websocket = None

    response = _FakeResponse(
        msgpack.packb({"encrypted_results": [b"server-c"], "elapsed_ms": 1.0}, use_bin_type=True)
    )
    protocol._http_session = _FakeSession(response=response)

    result = protocol._send_request(0, "qkv", ["cipher"])

    assert result == ["decoded:b'server-c'"]
    assert protocol._http_session.calls[0][1]["timeout"] == 123


def test_ensure_websocket_disables_keepalive_ping(monkeypatch):
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol._websocket = None
    protocol.websocket_url = "ws://server/api/layer/ws"
    protocol.auth_token = "token"

    captured = {}

    def fake_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "ws-conn"

    monkeypatch.setattr(layer_protocol_module, "connect", fake_connect)

    websocket = protocol._ensure_websocket()

    assert websocket == "ws-conn"
    assert captured["url"] == "ws://server/api/layer/ws"
    assert captured["kwargs"]["ping_interval"] is None
