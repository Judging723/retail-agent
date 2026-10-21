from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import AppException
from app.llm import LLMUnavailableError


def test_404_not_found(client: TestClient) -> None:
    response = client.get("/not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "not_found"
    request_id = data["error"]["request_id"]
    assert response.headers.get("x-request-id") == request_id


def test_business_app_exception(application: FastAPI, client: TestClient) -> None:
    @application.get("/test/app-error")
    def raise_app_error() -> None:
        raise AppException(
            code="order_conflict",
            message="Order state conflict",
            status_code=409,
        )

    response = client.get("/test/app-error")
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] == "order_conflict"
    assert data["error"]["message"] == "Order state conflict"


def test_validation_error(application: FastAPI, client: TestClient) -> None:
    @application.get("/test/items/{item_id}")
    def read_test_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    response = client.get("/test/items/not-an-integer")
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "validation_error"
    assert data["error"]["details"] is not None


def test_llm_unavailable_error(
    application: FastAPI,
    client: TestClient,
) -> None:
    @application.get("/test/llm-unavailable")
    def raise_llm_unavailable() -> None:
        raise LLMUnavailableError()

    response = client.get("/test/llm-unavailable")

    assert response.status_code == 503

    body = response.json()

    assert body["error"]["code"] == "llm_unavailable"
    assert body["error"]["message"] == ("Model service is temporarily unavailable")
    assert body["error"]["details"] is None
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]
