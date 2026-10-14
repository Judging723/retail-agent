from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "retail-agent-backend",
        "version": "0.1.0",
        "environment": "development",
    }
