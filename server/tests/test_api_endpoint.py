from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
import msgpack

from server import server
from server.handlers import inference_handler
from server.model import weight_manager
from server.services import layer_service
from server.model.weight_manager import ModelUnavailableError


def test_legacy_infer_route_registered():
    paths = {
        route.path
        for route in inference_handler.router.routes
        if isinstance(route, APIRoute)
    }
    assert "/api/infer" in paths


def test_layer_websocket_route_registered():
    ws_paths = {
        route.path
        for route in inference_handler.router.routes
        if isinstance(route, APIWebSocketRoute)
    }
    assert "/api/layer/ws" in ws_paths


def test_layer_websocket_route_processes_binary_payload(monkeypatch):
    app = FastAPI()
    app.include_router(inference_handler.router)

    captured = {}

    def fake_process_binary_payload(req_data, cid):
        captured["req_data"] = req_data
        captured["cid"] = cid
        return b"response-bytes"

    monkeypatch.setattr(inference_handler, "_process_binary_payload", fake_process_binary_payload)

    request_data = {"session_id": "s1", "layer_idx": 0, "operation": "qkv", "encrypted_vectors": []}

    with TestClient(app) as client:
        with client.websocket_connect("/api/layer/ws") as websocket:
            websocket.send_bytes(msgpack.packb(request_data, use_bin_type=True))
            assert websocket.receive_bytes() == b"response-bytes"

    assert captured["req_data"] == request_data
    assert isinstance(captured["cid"], str)


def test_get_health_returns_snet_payload_without_model_load(monkeypatch):
    called = False

    monkeypatch.setattr(
        server,
        "get_model_status",
        lambda: {
            "model": "test-model",
            "model_status": "not_loaded",
            "model_error": None,
        },
    )

    def fail_load_model():
        nonlocal called
        called = True
        raise AssertionError("health must not load the model")

    monkeypatch.setattr(server, "load_model", fail_load_model, raising=False)
    monkeypatch.setattr(weight_manager, "load_model", fail_load_model)

    with TestClient(server.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "zk-llm-turbo",
        "serviceID": "zk_llm1",
        "model": "test-model",
        "model_status": "not_loaded",
        "model_error": None,
    }
    assert called is False


def test_post_health_matches_publisher_proto_mapping(monkeypatch):
    monkeypatch.setattr(
        server,
        "get_model_status",
        lambda: {
            "model": "test-model",
            "model_status": "not_loaded",
            "model_error": None,
        },
    )

    with TestClient(server.app) as client:
        response = client.post("/health", json={"op": "health"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "zk-llm-turbo",
        "serviceID": "zk_llm1",
        "model": "test-model",
        "model_status": "not_loaded",
        "model_error": None,
    }


def test_root_route_returns_liveness_payload():
    with TestClient(server.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "zk-llm-turbo"}


def test_get_heartbeat_returns_snet_liveness_payload():
    with TestClient(server.app) as client:
        response = client.get("/heartbeat")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "zk-llm-turbo",
        "serviceID": "zk_llm1",
    }


def test_post_heartbeat_returns_snet_liveness_payload():
    with TestClient(server.app) as client:
        response = client.post("/heartbeat", json={"op": "heartbeat"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "zk-llm-turbo",
        "serviceID": "zk_llm1",
    }


def test_layer_returns_503_when_model_unavailable(monkeypatch):
    app = FastAPI()
    app.include_router(inference_handler.router)

    monkeypatch.setattr(layer_service, "get_session", lambda session_id: object())

    def unavailable(layer_idx):
        raise ModelUnavailableError("model download failed")

    monkeypatch.setattr(layer_service, "get_layer_weights", unavailable)

    with TestClient(app) as client:
        response = client.post(
            "/api/layer",
            json={
                "session_id": "session-1",
                "layer_idx": 0,
                "operation": "qkv",
                "encrypted_vectors_b64": [],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not available: model download failed"}
