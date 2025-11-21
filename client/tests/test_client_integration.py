import pytest, requests_mock, numpy as np
from client import client

@pytest.mark.integration
def test_client_post(monkeypatch):
    # Prevent network call to Hugging Face
    monkeypatch.setattr(client, "load_tokenizer", lambda: None)
    monkeypatch.setattr(client, "tokenize_prompt", lambda prompt, _: {"input_ids": [[0, 1, 2, 3]]})
    monkeypatch.setattr(client, "extract_embeddings", lambda _: np.random.rand(10, 2048))  # ✅ NumPy array mock

    # Mock server response
    with requests_mock.Mocker() as m:
        m.post("https://nonvocalic-stetson-unelderly.ngrok-free.dev/api/infer",
               json={"encrypted_result": "abc=="})
        monkeypatch.setattr(client, "load_server_config", lambda: {
            "base_url": "https://nonvocalic-stetson-unelderly.ngrok-free.dev",
            "infer_endpoint": "/api/infer",
            "auth_token": "dbskdjbvsjhdfjhsdgfkjsdgfkjsdgfkjsd"
        })

        # Mock TenSEAL vector deserialization
        mock_vector = type("MockVector", (), {"decrypt": lambda: [0.1, 0.2, 0.3]})
        monkeypatch.setattr(client.ts, "ckks_vector_from", lambda ctx, data: mock_vector)

        result = client.main("Integration test prompt")
        assert result is None  # ensure function completes

