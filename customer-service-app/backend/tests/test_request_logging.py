import logging
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_request_logging_generates_request_id(
    client: TestClient,
    caplog,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.request"):
        response = client.get("/api/v1/health")

    assert response.status_code == 200

    # HTTP 响应头不区分大小写
    request_id = response.headers["x-request-id"]

    # 如果不是合法 UUID，这行会直接抛异常
    UUID(request_id)

    assert "method=GET" in caplog.text
    assert "path=/api/v1/health" in caplog.text
    assert "status_code=200" in caplog.text


def test_request_logging_reuses_existing_request_id(
    client: TestClient,
) -> None:
    request_id = "frontend-request-001"

    response = client.get("/api/v1/health", headers={"x-request-id": request_id})

    assert response.headers["x-request-id"] == request_id


def test_request_logging_records_unhandled_exception(
    application: FastAPI,
    caplog,
) -> None:
    @application.get("/test-error")
    async def raise_error() -> None:
        raise RuntimeError("test error")

    request_id = "failed-request-001"

    with (
        TestClient(
            application,
            raise_server_exceptions=False,
        ) as test_client,
        caplog.at_level(logging.INFO, logger="app.request"),
    ):
        response = test_client.get(
            "/test-error",
            headers={"X-Request-ID": request_id},
        )

    assert response.status_code == 500
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "error": {
            "code": "internal_server_error",
            "message": "服务器内部错误",
            "request_id": request_id,
        }
    }
    assert "request_failed" in caplog.text
