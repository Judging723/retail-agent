from fastapi.testclient import TestClient

from app.core.config import get_settings


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    settings = get_settings()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "retail-agent-backend",
        "version": "0.1.0",
        "environment": settings.environment,
    }
