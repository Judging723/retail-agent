from uuid import UUID

from fastapi.testclient import TestClient


def test_request_logging_auto_generate_request_id(client: TestClient) -> None:
    """测试：未传入X-Request-ID，中间件自动生成合法UUID"""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert "x-request-id" in response.headers
    # 校验是合法UUID，格式不对会直接抛异常，测试失败
    request_id = response.headers["x-request-id"]
    UUID(request_id)


def test_request_logging_reuse_existing_request_id(client: TestClient) -> None:
    """测试：客户端传入X-Request-ID，后端原样保留透传"""
    request_id = "test-request-123"

    response = client.get(
        "/api/v1/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
