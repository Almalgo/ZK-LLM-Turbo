from client.inference.layer_protocol import EncryptedLayerProtocol


def test_build_websocket_url_from_http():
    assert (
        EncryptedLayerProtocol._build_websocket_url(
            "http://localhost:8000",
            "/api/layer/ws",
        )
        == "ws://localhost:8000/api/layer/ws"
    )


def test_build_websocket_url_from_https():
    assert (
        EncryptedLayerProtocol._build_websocket_url(
            "https://example.com",
            "/api/layer/ws",
        )
        == "wss://example.com/api/layer/ws"
    )
