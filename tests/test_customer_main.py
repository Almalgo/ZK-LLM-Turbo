from customer_main import run


def test_health_returns_ok_without_inference():
    result = run({"op": "health"})

    assert result == {"serviceID": "zk_llm1", "status": "SERVING"}


def test_heartbeat_alias_returns_snet_serving_status():
    result = run({"op": "heartbeat"})

    assert result == {"serviceID": "zk_llm1", "status": "SERVING"}


def test_missing_op_returns_structured_error():
    assert run({}) == {"error": "op is required", "error_type": "InvalidInput"}


def test_unknown_op_returns_structured_error():
    assert run({"op": "unknown"}) == {
        "error": "Unsupported op: unknown",
        "error_type": "InvalidInput",
    }


def test_existing_layer_operation_shape_dispatches_to_layer(monkeypatch):
    captured = {}

    def fake_process_layer_request(input_data):
        captured["input_data"] = input_data
        return {"ok": True}

    monkeypatch.setattr("customer_main._process_layer_request", fake_process_layer_request)

    payload = {
        "session_id": "s1",
        "layer_idx": 0,
        "operation": "qkv",
        "encrypted_vectors_b64": [],
    }

    assert run(payload) == {"ok": True}
    assert captured["input_data"] == payload
