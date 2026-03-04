import base64

import pytest
import requests_mock

from client import client
from client.encryption.ckks_context import create_ckks_context, serialize_public_context


@pytest.mark.integration
def test_setup_session_posts_public_context_and_returns_session_id():
    context = create_ckks_context()
    public_b64 = base64.b64encode(serialize_public_context(context)).decode("utf-8")

    server_cfg = {
        "base_url": "http://localhost:8000",
        "session_endpoint": "/api/session",
        "auth_token": "test-token",
    }

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8000/api/session",
            json={"session_id": "sess-123"},
            status_code=200,
        )

        session_id = client.setup_session(context, server_cfg)

    assert session_id == "sess-123"
    assert m.call_count == 1
    sent_body = m.last_request.json()
    assert sent_body["public_context_b64"] == public_b64
