import pytest
import time
from fastapi.testclient import TestClient
from client.encryption.ckks_context import create_ckks_context, serialize_public_context
from server.handlers import session_handler
from server.server import app
import base64


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    session_handler._sessions.clear()
    session_handler.load_session_config.cache_clear()
    yield
    session_handler._sessions.clear()
    session_handler.load_session_config.cache_clear()


def _public_context_b64() -> str:
    context = create_ckks_context()
    return base64.b64encode(serialize_public_context(context)).decode("utf-8")


def test_session_expires_after_ttl(monkeypatch):
    context = create_ckks_context()
    session_handler._sessions["expire-test"] = session_handler.SessionEntry(
        context=context,
        created_at=100.0,
        last_accessed=100.0,
    )

    removed = session_handler.cleanup_expired_sessions(max_age_seconds=50, now=200.0)
    assert removed == 1
    assert "expire-test" not in session_handler._sessions


def test_get_session_still_valid(monkeypatch):
    context = create_ckks_context()
    session_handler._sessions["valid"] = session_handler.SessionEntry(
        context=context,
        created_at=100.0,
        last_accessed=150.0,
    )

    removed = session_handler.cleanup_expired_sessions(max_age_seconds=50, now=200.0)
    assert removed == 0
    assert "valid" in session_handler._sessions


def test_session_not_accessed_updates_timestamp(monkeypatch):
    context = create_ckks_context()
    entry = session_handler.SessionEntry(
        context=context,
        created_at=100.0,
        last_accessed=100.0,
    )
    session_handler._sessions["test"] = entry

    monkeypatch.setattr(session_handler.time, "time", lambda: 200.0)
    session_handler.get_session("test")

    assert session_handler._sessions["test"].last_accessed == 200.0