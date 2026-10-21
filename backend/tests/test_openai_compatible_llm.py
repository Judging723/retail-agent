from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm import (
    ChatMessage,
    ChatRole,
    LLMInvalidResponseError,
    OpenAICompatibleLLMClient,
)

pytestmark = pytest.mark.anyio


async def test_openai_compatible_client_maps_chat_response() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            model="served-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  您好，请提供订单编号。  "),
                    finish_reason="stop",
                )
            ],
        )
    )
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    client = OpenAICompatibleLLMClient(
        client=sdk_client,
        model="configured-model",
    )

    response = await client.chat(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="你是一名电商客服。",
            ),
            ChatMessage(
                role=ChatRole.USER,
                content="帮我查询订单。",
            ),
        ]
    )

    assert response.content == "您好，请提供订单编号。"
    assert response.model == "served-model"
    assert response.finish_reason == "stop"

    create.assert_awaited_once_with(
        model="configured-model",
        messages=[
            {
                "role": "system",
                "content": "你是一名电商客服。",
            },
            {
                "role": "user",
                "content": "帮我查询订单。",
            },
        ],
    )

    async def test_openai_compatible_client_rejects_empty_choices() -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                model="served-model",
                choices=[],
            )
        )
        sdk_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        client = OpenAICompatibleLLMClient(
            client=sdk_client,
            model="configured-model",
        )

        with pytest.raises(LLMInvalidResponseError):
            await client.chat(
                messages=[
                    ChatMessage(
                        role=ChatRole.USER,
                        content="你好。",
                    )
                ]
            )
