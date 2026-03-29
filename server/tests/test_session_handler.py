import base64

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from client.encryption.ckks_context import create_ckks_context, serialize_public_context
from server.handlers import session_handler
from server.server import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    session_handler._sessions.clear()
    yield
    session_handler._sessions.clear()


def _public_context_b64() -> str:
    context = create_ckks_context()
    return base64.b64encode(serialize_public_context(context)).decode("utf-8")


def test_cleanup_expired_sessions_removes_idle_entries():
    context = create_ckks_context()
    session_handler._sessions["expired"] = session_handler.SessionEntry(
        context=context,
        created_at=100.0,
        last_accessed=100.0,
    )
    session_handler._sessions["active"] = session_handler.SessionEntry(
        context=context,
        created_at=180.0,
        last_accessed=195.0,
    )

    removed = session_handler.cleanup_expired_sessions(max_age_seconds=60, now=200.0)

    assert removed == 1
    assert "expired" not in session_handler._sessions
    assert "active" in session_handler._sessions


def test_get_session_updates_last_accessed(monkeypatch):
    context = create_ckks_context()
    session_handler._sessions["abc"] = session_handler.SessionEntry(
        context=context,
        created_at=10.0,
        last_accessed=20.0,
    )

    monkeypatch.setattr(session_handler.time, "time", lambda: 123.0)
    returned = session_handler.get_session("abc")

    assert returned is context
    assert session_handler._sessions["abc"].last_accessed == 123.0


def test_get_session_404_after_cleanup():
    context = create_ckks_context()
    session_handler._sessions["expired"] = session_handler.SessionEntry(
        context=context,
        created_at=0.0,
        last_accessed=0.0,
    )
    session_handler.cleanup_expired_sessions(max_age_seconds=1, now=5.0)

    with pytest.raises(HTTPException) as exc:
        session_handler.get_session("expired")

    assert exc.value.status_code == 404


def test_cleanup_excess_sessions_evicts_lru():
    context = create_ckks_context()
    session_handler._sessions["oldest"] = session_handler.SessionEntry(
        context=context,
        created_at=0.0,
        last_accessed=10.0,
    )
    session_handler._sessions["newer"] = session_handler.SessionEntry(
        context=context,
        created_at=0.0,
        last_accessed=20.0,
    )

    removed = session_handler.cleanup_excess_sessions(max_sessions=1)

    assert removed == 1
    assert "oldest" not in session_handler._sessions
    assert "newer" in session_handler._sessions


def test_delete_session_endpoint():
    response = client.post("/api/session", json={"public_context_b64": _public_context_b64()})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    delete_response = client.delete(f"/api/session/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"session_id": session_id, "deleted": True}

    missing_response = client.delete(f"/api/session/{session_id}")
    assert missing_response.status_code == 404
