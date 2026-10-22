from collections.abc import Sequence
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from app.api.dependencies import get_llm_client
from app.llm import ChatMessage, LLMUnavailableError

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        return self.response


class FailingLLMClient:
    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise LLMUnavailableError("模拟模型不可用")


async def test_multi_turn_chat_flow(
    application: FastAPI,
    api_client: AsyncClient,
) -> None:
    fake_llm = FakeLLMClient("请提供您的订单编号。")
    application.dependency_overrides[get_llm_client] = lambda: fake_llm

    try:
        suffix = uuid4().hex[:12]
        create_response = await api_client.post(
            "/api/v1/conversations",
            json={
                "customer_no": f"CUST-CHAT-{suffix}",
                "display_name": "聊天测试用户",
            },
        )
        assert create_response.status_code == 201
        conversation_no = create_response.json()["conversation_no"]

        first_response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat",
            json={"content": "帮我查询物流。"},
        )
        second_response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat",
            json={"content": "订单号是 123456。"},
        )
        history_response = await api_client.get(f"/api/v1/conversations/{conversation_no}")
    finally:
        application.dependency_overrides.pop(get_llm_client, None)

    assert first_response.status_code == 201
    assert first_response.json()["sequence_no"] == 2
    assert first_response.json()["role"] == "assistant"

    assert second_response.status_code == 201
    assert second_response.json()["sequence_no"] == 4

    assert history_response.status_code == 200
    messages = history_response.json()["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["content"] for message in messages] == [
        "帮我查询物流。",
        "请提供您的订单编号。",
        "订单号是 123456。",
        "请提供您的订单编号。",
    ]

    assert [message["role"] for message in fake_llm.calls[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


async def test_same_customer_can_create_multiple_conversations(
    api_client: AsyncClient,
) -> None:
    customer_no = f"CUST-MULTI-{uuid4().hex[:12]}"
    payload = {"customer_no": customer_no, "display_name": "重复会话测试用户"}

    first_response = await api_client.post("/api/v1/conversations", json=payload)
    second_response = await api_client.post("/api/v1/conversations", json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["conversation_no"] != second_response.json()["conversation_no"]


async def test_get_missing_conversation_returns_404(
    api_client: AsyncClient,
) -> None:
    response = await api_client.get("/api/v1/conversations/CONV-00000000000000000000")

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


async def test_failed_llm_call_does_not_save_messages(
    application: FastAPI,
    api_client: AsyncClient,
) -> None:
    failing_llm = FailingLLMClient()
    application.dependency_overrides[get_llm_client] = lambda: failing_llm

    try:
        suffix = uuid4().hex[:12]

        create_response = await api_client.post(
            "/api/v1/conversations",
            json={
                "customer_no": f"CUST-FAIL-{suffix}",
                "display_name": "失败事务测试用户",
            },
        )
        conversation_no = create_response.json()["conversation_no"]

        chat_response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat",
            json={"content": "帮我查询订单。"},
        )

        history_response = await api_client.get(f"/api/v1/conversations/{conversation_no}")
    finally:
        application.dependency_overrides.pop(get_llm_client, None)

    assert chat_response.status_code == 503
    assert chat_response.json() == {"detail": "模拟模型不可用"}

    assert history_response.status_code == 200
    assert history_response.json()["messages"] == []
