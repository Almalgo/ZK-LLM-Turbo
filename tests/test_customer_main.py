import requests

from customer_main import run


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


def test_health_returns_minimal_haas_serving_status(monkeypatch):
    monkeypatch.delenv("ZKLLM_BACKEND_BASE_URL", raising=False)
    monkeypatch.setattr("customer_main.DEFAULT_BACKEND_BASE_URL", "")

    result = run({"op": "health"})

    assert result == {"serviceID": "zk_llm2", "status": "SERVING"}


def test_heartbeat_alias_returns_proxy_health(monkeypatch):
    monkeypatch.delenv("ZKLLM_BACKEND_BASE_URL", raising=False)
    monkeypatch.setattr("customer_main.DEFAULT_BACKEND_BASE_URL", "")

    result = run({"op": "heartbeat"})

    assert result == {"serviceID": "zk_llm2", "status": "SERVING"}


def test_proxy_health_without_backend_configured_stays_serving(monkeypatch):
    monkeypatch.delenv("ZKLLM_BACKEND_BASE_URL", raising=False)
    monkeypatch.setattr("customer_main.DEFAULT_BACKEND_BASE_URL", "")

    result = run({"op": "proxy_health"})

    assert result["serviceID"] == "zk_llm2"
    assert result["status"] == "SERVING"
    assert result["mode"] == "proxy"
    assert result["backend"] == {
        "configured": False,
        "reachable": False,
        "base_url": None,
        "status_code": None,
        "error": None,
    }


def test_health_with_reachable_backend(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")

    def fake_get(url, headers, timeout):
        assert url == "https://backend.example/health"
        assert headers == {"Content-Type": "application/json"}
        assert timeout == 10.0
        return FakeResponse(status_code=200, json_data={"status": "ok"})

    monkeypatch.setattr("customer_main.requests.get", fake_get)

    result = run({"op": "proxy_health"})

    assert result["status"] == "SERVING"
    assert result["backend"]["configured"] is True
    assert result["backend"]["reachable"] is True
    assert result["backend"]["base_url"] == "https://backend.example"
    assert result["backend"]["status_code"] == 200
    assert result["backend"]["error"] is None


def test_health_with_backend_timeout_stays_serving(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")

    def fake_get(url, headers, timeout):
        raise requests.Timeout("timeout")

    monkeypatch.setattr("customer_main.requests.get", fake_get)

    result = run({"op": "proxy_health"})

    assert result["status"] == "SERVING"
    assert result["backend"]["configured"] is True
    assert result["backend"]["reachable"] is False
    assert "timeout" in result["backend"]["error"]


def test_health_can_fail_closed_when_configured(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")
    monkeypatch.setenv("ZKLLM_PROXY_FAIL_OPEN_HEALTH", "false")

    def fake_get(url, headers, timeout):
        raise requests.Timeout("timeout")

    monkeypatch.setattr("customer_main.requests.get", fake_get)

    result = run({"op": "proxy_health"})

    assert result["status"] == "UNAVAILABLE"
    assert result["backend"]["reachable"] is False


def test_missing_op_returns_structured_error():
    assert run({}) == {"error": "op is required", "error_type": "InvalidInput"}


def test_unknown_op_returns_structured_error():
    assert run({"op": "unknown"}) == {
        "error": "Unsupported op: unknown",
        "error_type": "InvalidInput",
    }


def test_session_requires_public_context(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")

    assert run({"op": "session"}) == {
        "error": "public_context_b64 is required",
        "error_type": "InvalidInput",
    }


def test_session_requires_backend_url(monkeypatch):
    monkeypatch.delenv("ZKLLM_BACKEND_BASE_URL", raising=False)
    monkeypatch.setattr("customer_main.DEFAULT_BACKEND_BASE_URL", "")

    result = run({"op": "session", "public_context_b64": "ctx"})

    assert result == {
        "error": "ZKLLM_BACKEND_BASE_URL is required for proxy operation",
        "error_type": "BackendNotConfigured",
    }


def test_session_forwards_to_backend_with_bearer_token(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example/")
    monkeypatch.setenv("ZKLLM_BACKEND_AUTH_TOKEN", "secret-token")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(status_code=200, json_data={"session_id": "session-1"})

    monkeypatch.setattr("customer_main.requests.post", fake_post)

    payload = {"op": "session", "public_context_b64": "ctx"}
    assert run(payload) == {"session_id": "session-1"}
    assert captured == {
        "url": "https://backend.example/api/session",
        "json": payload,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-token",
        },
        "timeout": 900.0,
    }


def test_layer_forwards_to_backend(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(
            status_code=200,
            json_data={
                "encrypted_results_b64": ["result"],
                "operation": "qkv",
                "layer_idx": 0,
                "elapsed_ms": 1.2,
            },
        )

    monkeypatch.setattr("customer_main.requests.post", fake_post)

    payload = {
        "op": "layer",
        "session_id": "s1",
        "layer_idx": 0,
        "operation": "qkv",
        "encrypted_vectors_b64": ["vec"],
    }
    result = run(payload)

    assert result["encrypted_results_b64"] == ["result"]
    assert captured["url"] == "https://backend.example/api/layer"
    assert captured["json"] == payload


def test_layer_operation_alias_maps_to_layer_proxy(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(status_code=200, json_data={"ok": True})

    monkeypatch.setattr("customer_main.requests.post", fake_post)

    payload = {
        "operation": "qkv",
        "session_id": "s1",
        "layer_idx": 0,
        "encrypted_vectors_b64": ["vec"],
    }
    assert run(payload) == {"ok": True}
    assert captured["url"] == "https://backend.example/api/layer"
    assert captured["json"] == payload


def test_backend_non_2xx_returns_structured_error(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")

    def fake_post(url, json, headers, timeout):
        return FakeResponse(status_code=503, text="unavailable")

    monkeypatch.setattr("customer_main.requests.post", fake_post)

    result = run({"op": "layer", "operation": "qkv"})

    assert result == {
        "error": "Backend returned HTTP 503",
        "error_type": "BackendHTTPError",
        "status_code": 503,
        "backend_body": "unavailable",
    }


def test_backend_request_exception_returns_structured_error(monkeypatch):
    monkeypatch.setenv("ZKLLM_BACKEND_BASE_URL", "https://backend.example")

    def fake_post(url, json, headers, timeout):
        raise requests.ConnectionError("connection failed")

    monkeypatch.setattr("customer_main.requests.post", fake_post)

    result = run({"op": "layer", "operation": "qkv"})

    assert result["error_type"] == "BackendRequestError"
    assert "connection failed" in result["error"]
