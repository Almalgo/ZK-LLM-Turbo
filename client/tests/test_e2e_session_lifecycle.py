import pytest
import base64
from fastapi.testclient import TestClient
from client.encryption.ckks_context import create_ckks_context, serialize_public_context
from server.handlers import session_handler
from server.server import app


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


def test_session_lifecycle_create_and_delete():
    create_response = client.post("/api/session", json={"public_context_b64": _public_context_b64()})
    assert create_response.status_code == 200
    session_id = create_response.json()["session_id"]

    get_response = client.get(f"/api/session/{session_id}")
    assert get_response.status_code == 200

    delete_response = client.delete(f"/api/session/{session_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_session_lifecycle_double_delete_fails():
    create_response = client.post("/api/session", json={"public_context_b64": _public_context_b64()})
    session_id = create_response.json()["session_id"]

    client.delete(f"/api/session/{session_id}")
    second_delete = client.delete(f"/api/session/{session_id}")
    assert second_delete.status_code == 404


def test_session_lifecycle_get_after_delete_fails():
    create_response = client.post("/api/session", json={"public_context_b64": _public_context_b64()})
    session_id = create_response.json()["session_id"]

    client.delete(f"/api/session/{session_id}")
    get_response = client.get(f"/api/session/{session_id}")
    assert get_response.status_code == 404