from fastapi.routing import APIRoute, APIWebSocketRoute

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
