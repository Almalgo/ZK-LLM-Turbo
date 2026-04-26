import pytest
from unittest.mock import Mock, patch
from client.inference.layer_protocol import EncryptedLayerProtocol


class MockModelConfig:
    num_hidden_layers = 22
    hidden_size = 2048
    num_attention_heads = 32
    num_key_value_heads = 4
    rms_norm_eps = 1e-5


def test_protocol_init_with_timeouts():
    protocol = EncryptedLayerProtocol(
        context=None,
        session_id="test-session",
        server_url="http://localhost:8000",
        layer_endpoint="/api/layer",
        auth_token="test_token",
        model_config=MockModelConfig(),
        use_websocket=True,
        websocket_open_timeout=60,
        websocket_close_timeout=20,
    )
    assert protocol.websocket_open_timeout == 60
    assert protocol.websocket_close_timeout == 20


def test_protocol_default_timeouts():
    protocol = EncryptedLayerProtocol(
        context=None,
        session_id="test-session",
        server_url="http://localhost:8000",
        layer_endpoint="/api/layer",
        auth_token="test_token",
        model_config=MockModelConfig(),
        use_websocket=True,
    )
    assert protocol.websocket_open_timeout == 30
    assert protocol.websocket_close_timeout == 10


def test_protocol_reset_round_metrics_clears_data():
    protocol = EncryptedLayerProtocol(
        context=None,
        session_id="test-session",
        server_url="http://localhost:8000",
        layer_endpoint="/api/layer",
        auth_token="test_token",
        model_config=MockModelConfig(),
    )
    protocol._round_metrics.append({"op": "test", "roundtrip_ms": 100})
    protocol.reset_round_metrics()
    assert len(protocol._round_metrics) == 0


def test_protocol_get_round_metrics_returns_list():
    protocol = EncryptedLayerProtocol(
        context=None,
        session_id="test-session",
        server_url="http://localhost:8000",
        layer_endpoint="/api/layer",
        auth_token="test_token",
        model_config=MockModelConfig(),
    )
    protocol._round_metrics.append({"op": "qkv", "roundtrip_ms": 50})
    protocol._round_metrics.append({"op": "o_proj", "roundtrip_ms": 30})
    metrics = protocol.get_round_metrics()
    assert len(metrics) == 2
    assert metrics[0]["op"] == "qkv"