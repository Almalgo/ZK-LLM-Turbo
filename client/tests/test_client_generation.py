from types import SimpleNamespace

import numpy as np
import torch

from client import client


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, list):
            return "".join(f"t{token_id}" for token_id in token_ids)
        return ""


class _FakeLayer:
    def __init__(self, dtype=torch.float32, return_tuple=True):
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.calls = []
        self.return_tuple = return_tuple

    def parameters(self):
        yield self._param

    def __call__(self, hidden_t, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("position_embeddings") is not None
        assert hidden_t.dtype == self._param.dtype
        if self.return_tuple:
            return (hidden_t,)
        return hidden_t


def test_generate_passes_position_embeddings_to_plaintext_layers(monkeypatch):
    fake_layer0 = _FakeLayer(dtype=torch.float16)
    fake_layer1 = _FakeLayer(dtype=torch.float16, return_tuple=False)

    model_config = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=1,
        num_attention_heads=1,
        hidden_size=4,
        rms_norm_eps=1e-5,
    )
    fake_model = SimpleNamespace(
        model=SimpleNamespace(
            rotary_emb=lambda hidden_t, position_ids: (
                torch.zeros_like(hidden_t),
                torch.zeros_like(hidden_t),
            )
        )
    )

    monkeypatch.setattr(
        client,
        "load_config",
        lambda: (
            {"inference": {"use_websocket": False, "use_async_pipeline": False}},
            {
                "base_url": "http://server",
                "layer_endpoint": "/api/layer",
                "auth_token": "token",
                "layer_ws_endpoint": "/api/layer/ws",
                "session_endpoint": "/api/session",
            },
        ),
    )
    monkeypatch.setattr(client, "load_tokenizer", lambda: _FakeTokenizer())
    monkeypatch.setattr(
        client,
        "tokenize_prompt",
        lambda prompt, tokenizer: {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
    )
    monkeypatch.setattr(
        client,
        "get_model_components",
        lambda: {
            "model": fake_model,
            "layers": [fake_layer0, fake_layer1],
            "final_norm_weight": np.ones(4, dtype=np.float32),
            "lm_head_weight": np.ones((8, 4), dtype=np.float32),
            "config": model_config,
        },
    )
    monkeypatch.setattr(client, "create_ckks_context", lambda: object())
    monkeypatch.setattr(client, "get_encrypted_layer_params", lambda *args, **kwargs: [])
    monkeypatch.setattr(client, "setup_session", lambda context, server_cfg: "session")
    monkeypatch.setattr(client, "extract_embeddings", lambda input_ids: np.zeros((2, 4), dtype=np.float32))
    monkeypatch.setattr(
        client,
        "run_encrypted_layers",
        lambda hidden_states, protocol, encrypted_layer_params, position_offset: hidden_states,
    )

    class _FakeProtocol:
        def close(self):
            return None

    monkeypatch.setattr(client, "EncryptedLayerProtocol", lambda **kwargs: _FakeProtocol())

    result = client.generate(
        prompt="hello",
        num_tokens=1,
        num_encrypted_layers=1,
        quiet=True,
        return_stats=True,
        show_stats=False,
    )

    assert result.tokens_generated == 1
    assert len(fake_layer1.calls) == 1
