from fastapi.testclient import TestClient
from server.model.weight_manager import MODEL_NAME
from server.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_before_startup():
    app.state.model_ready = False
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "starting"}


def test_ready_endpoint_when_model_ready():
    app.state.model_ready = True
    app.state.model_name = MODEL_NAME
    try:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "model": MODEL_NAME,
        }
    finally:
        app.state.model_ready = False


def test_infer_endpoint():
    response = client.post("/api/infer", json={"encrypted_embeddings": [], "metadata": {}})
    assert response.status_code in (200, 422)
