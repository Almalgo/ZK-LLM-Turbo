import pytest
from unittest.mock import patch
import requests


def test_network_error_on_session_setup():
    with patch("client.client._http_session.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")
        from client.client import load_config
        import os
        os.environ["ZKLLM_SERVER_AUTH_TOKEN"] = "fake_token"
        client_cfg, server_cfg = load_config()
        from client.client import setup_session
        with pytest.raises(requests.exceptions.ConnectionError):
            setup_session(client_cfg.get("ckks"), server_cfg)


def test_network_timeout_on_session_setup():
    with patch("client.client._http_session.post") as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Timeout")
        from client.client import load_config
        import os
        os.environ["ZKLLM_SERVER_AUTH_TOKEN"] = "fake_token"
        client_cfg, server_cfg = load_config()
        from client.client import setup_session
        with pytest.raises(requests.exceptions.Timeout):
            setup_session(client_cfg.get("ckks"), server_cfg)


def test_invalid_credentials_rejected():
    from server.security import validate_bearer_token
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        validate_bearer_token(authorization="Bearer wrong_token")
    assert exc.value.status_code == 403


def test_missing_auth_header_rejected():
    from server.security import validate_bearer_token
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        validate_bearer_token(authorization=None)
    assert exc.value.status_code == 401
