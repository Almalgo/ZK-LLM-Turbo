from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
import msgpack

from server.handlers import inference_handler


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
