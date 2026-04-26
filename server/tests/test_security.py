import os
import pytest
from server.security import _auth_required, _get_required_token, _is_secure_env_enabled


def test_auth_disabled_when_no_token(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ZKLLM_API_TOKEN", raising=False)
    monkeypatch.delenv("ZKLLM_SERVER_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ZKLLM_REQUIRE_API_TOKEN", "false")
    assert _get_required_token() == ""


def test_auth_enabled_by_default(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test_token")
    monkeypatch.delenv("ZKLLM_REQUIRE_API_TOKEN", raising=False)
    assert _auth_required() is True


def test_auth_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test_token")
    monkeypatch.setenv("ZKLLM_REQUIRE_API_TOKEN", "false")
    assert _auth_required() is False


def test_secure_env_enabled_when_token_set(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test_token")
    monkeypatch.delenv("ZKLLM_REQUIRE_API_TOKEN", raising=False)
    assert _is_secure_env_enabled() is True


def test_secure_env_disabled_when_no_token(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ZKLLM_API_TOKEN", raising=False)
    monkeypatch.delenv("ZKLLM_SERVER_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ZKLLM_REQUIRE_API_TOKEN", "false")
    assert _is_secure_env_enabled() is False


def test_server_auth_token_alias(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ZKLLM_API_TOKEN", raising=False)
    monkeypatch.setenv("ZKLLM_SERVER_AUTH_TOKEN", "server-token")
    assert _get_required_token() == "server-token"
