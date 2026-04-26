import os

from client.client import load_config


def test_load_config_applies_server_env_overrides(monkeypatch):
    monkeypatch.setenv("ZKLLM_SERVER_BASE_URL", "http://127.0.0.1:8001")
    monkeypatch.setenv("ZKLLM_SERVER_LAYER_WS_ENDPOINT", "/custom/ws")
    monkeypatch.setenv("ZKLLM_REQUEST_TIMEOUT_SECONDS", "123")
    monkeypatch.setenv("ZKLLM_SERVER_AUTH_TOKEN", "tests-token")

    client_cfg, server_cfg = load_config()

    assert server_cfg["base_url"] == "http://127.0.0.1:8001"
    assert server_cfg["layer_ws_endpoint"] == "/custom/ws"
    assert client_cfg["inference"]["request_timeout_seconds"] == 123.0

    monkeypatch.delenv("ZKLLM_SERVER_BASE_URL", raising=False)
    monkeypatch.delenv("ZKLLM_SERVER_LAYER_WS_ENDPOINT", raising=False)
    monkeypatch.delenv("ZKLLM_REQUEST_TIMEOUT_SECONDS", raising=False)
