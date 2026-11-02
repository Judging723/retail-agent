from uuid import uuid4

import pytest
from httpx2 import AsyncClient

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_register_login_and_me(api_client: AsyncClient) -> None:
    customer_no = f"CUST-AUTH-{uuid4().hex[:12]}"
    password = "correct-password-123"

    register_response = await api_client.post(
        "/api/v1/auth/register",
        json={
            "customer_no": customer_no,
            "display_name": "认证测试用户",
            "password": password,
        },
    )
    login_response = await api_client.post(
        "/api/v1/auth/login",
        json={"customer_no": customer_no, "password": password},
    )
    token = login_response.json()["access_token"]
    me_response = await api_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert register_response.status_code == 201
    assert register_response.json()["token_type"] == "bearer"
    assert login_response.status_code == 200
    assert me_response.status_code == 200
    assert me_response.json() == {
        "customer_no": customer_no,
        "display_name": "认证测试用户",
    }


async def test_invalid_credentials_and_duplicate_registration(api_client: AsyncClient) -> None:
    customer_no = f"CUST-AUTH-DUP-{uuid4().hex[:12]}"
    payload = {
        "customer_no": customer_no,
        "display_name": None,
        "password": "correct-password-123",
    }

    first = await api_client.post("/api/v1/auth/register", json=payload)
    duplicate = await api_client.post("/api/v1/auth/register", json=payload)
    invalid_login = await api_client.post(
        "/api/v1/auth/login",
        json={"customer_no": customer_no, "password": "incorrect-password"},
    )

    assert first.status_code == 201
    assert duplicate.status_code == 409
    assert invalid_login.status_code == 401


async def test_conversations_require_authentication(api_client: AsyncClient) -> None:
    response = await api_client.post("/api/v1/conversations")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_customer_cannot_read_another_customers_conversation(
    api_client: AsyncClient,
    register_customer,
) -> None:
    first_headers = await register_customer(f"CUST-OWNER-{uuid4().hex[:12]}")
    created = await api_client.post("/api/v1/conversations", headers=first_headers)
    conversation_no = created.json()["conversation_no"]

    second_headers = await register_customer(f"CUST-OTHER-{uuid4().hex[:12]}")
    response = await api_client.get(
        f"/api/v1/conversations/{conversation_no}",
        headers=second_headers,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
