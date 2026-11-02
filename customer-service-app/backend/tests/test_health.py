import pytest
from fastapi.testclient import TestClient
from httpx2 import AsyncClient

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


@pytest.mark.anyio
@pytest.mark.integration
async def test_readiness_reports_database_and_lazy_rag(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "components": {
            "database": "ready",
            "rag": "lazy",
        },
    }
