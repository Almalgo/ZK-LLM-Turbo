import requests

from client import client


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_setup_session_success(monkeypatch):
    monkeypatch.setattr(client, "serialize_public_context", lambda context: b"ctx")

    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, timeout))
        return _Response({"session_id": "sid-1"})

    monkeypatch.setattr(client._http_session, "post", fake_post)

    session_id = client.setup_session(
        context=object(),
        server_cfg={
            "base_url": "http://server",
            "session_endpoint": "/api/session",
            "auth_token": "token",
        },
    )

    assert session_id == "sid-1"
    assert calls == [("http://server/api/session", 30)]


def test_setup_session_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(client, "serialize_public_context", lambda context: b"ctx")
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(client.random, "uniform", lambda a, b: 0.0)

    attempts = {"count": 0}

    def fake_post(url, json, headers, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.exceptions.ConnectionError("boom")
        return _Response({"session_id": "sid-2"})

    monkeypatch.setattr(client._http_session, "post", fake_post)

    session_id = client.setup_session(
        context=object(),
        server_cfg={
            "base_url": "http://server",
            "session_endpoint": "/api/session",
            "auth_token": "token",
            "session_setup_max_attempts": 3,
            "session_setup_retry_delay_seconds": 0.01,
        },
    )

    assert session_id == "sid-2"
    assert attempts["count"] == 2
