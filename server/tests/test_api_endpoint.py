from fastapi.testclient import TestClient
from server import server

client = TestClient(app)

def test_infer_endpoint():
    response = client.post("/api/infer", json={"encrypted_embeddings": [], "metadata": {}})
    assert response.status_code in (200, 422)  # Expect 422 due to missing data
