import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from app.api.dependencies import get_llm_client
from app.llm import (
    ChatResponse,
    ChatRole,
    FakeLLMClient,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_agent_reply_endpoint(
    application: FastAPI,
    api_client: AsyncClient,
) -> None:
    fake_llm = FakeLLMClient(
        ChatResponse(
            content="请提供您的订单编号。",
            model="fake-model",
            finish_reason="stop",
        )
    )

    def override_llm_client() -> FakeLLMClient:
        return fake_llm

    application.dependency_overrides[get_llm_client] = override_llm_client

    try:
        create_response = await api_client.post(
            "/api/v1/conversations",
            json={
                "customer_no": "CUST-AGENT-API-001",
                "display_name": "Agent API 测试用户",
            },
        )
        assert create_response.status_code == 201

        conversation_no = create_response.json()["conversation_no"]

        response = await api_client.post(
            (f"/api/v1/conversations/{conversation_no}/agent-replies"),
            json={
                "content": "帮我查询物流。",
            },
        )
    finally:
        application.dependency_overrides.pop(
            get_llm_client,
            None,
        )

    assert response.status_code == 201

    body = response.json()

    assert body["conversation_no"] == conversation_no
    assert body["sequence_no"] == 2
    assert body["role"] == "assistant"
    assert body["content"] == "请提供您的订单编号。"

    assert len(fake_llm.calls) == 1
    assert [message.role for message in fake_llm.calls[0]] == [
        ChatRole.SYSTEM,
        ChatRole.USER,
    ]


async def test_agent_reply_returns_503_without_llm(
    api_client: AsyncClient,
) -> None:
    create_response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": "CUST-NO-LLM-001",
            "display_name": "未配置模型测试用户",
        },
    )
    assert create_response.status_code == 201

    conversation_no = create_response.json()["conversation_no"]

    response = await api_client.post(
        (f"/api/v1/conversations/{conversation_no}/agent-replies"),
        json={
            "content": "你好。",
        },
    )

    assert response.status_code == 503

    body = response.json()

    assert body["error"]["code"] == "llm_unavailable"
    assert body["error"]["details"] is None
