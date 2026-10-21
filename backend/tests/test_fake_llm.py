import pytest

from app.llm import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    FakeLLMClient,
)

pytestmark = pytest.mark.anyio


async def test_fake_llm_returns_response_and_records_call() -> None:
    configured_response = ChatResponse(
        content="您好，请提供订单编号。",
        model="fake-model",
        finish_reason="stop",
    )
    client = FakeLLMClient(configured_response)

    messages = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="你是一名电商客服。",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="帮我查询物流。",
        ),
    ]

    response = await client.chat(messages=messages)

    assert response == configured_response
    assert client.calls == [tuple(messages)]
