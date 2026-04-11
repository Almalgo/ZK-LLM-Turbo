import torch

from client.model import embedding_extractor
from server.model import weight_manager


def test_client_model_dtype_env_override(monkeypatch):
    monkeypatch.setenv("ZKLLM_CLIENT_MODEL_DTYPE", "fp16")
    assert embedding_extractor._resolve_model_dtype() == torch.float16


def test_server_model_dtype_env_override(monkeypatch):
    monkeypatch.setenv("ZKLLM_SERVER_MODEL_DTYPE", "bf16")
    assert weight_manager._resolve_model_dtype() == torch.bfloat16


def test_model_dtype_invalid_value_falls_back_to_fp32(monkeypatch):
    monkeypatch.setenv("ZKLLM_CLIENT_MODEL_DTYPE", "not-a-dtype")
    monkeypatch.setenv("ZKLLM_SERVER_MODEL_DTYPE", "not-a-dtype")
    assert embedding_extractor._resolve_model_dtype() == torch.float32
    assert weight_manager._resolve_model_dtype() == torch.float32
