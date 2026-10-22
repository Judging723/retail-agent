from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm import LLMUnavailableError, OpenAICompatibleLLMClient

pytestmark = pytest.mark.anyio


async def test_openai_compatible_client_returns_text() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  您好！  "))]
        )
    )
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(
        client=sdk_client,
        model="test-model",
    )

    result = await client.chat([{"role": "user", "content": "你好"}])

    assert result == "您好！"
    create.assert_awaited_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "你好"}],
    )


async def test_openai_compatible_client_rejects_empty_response() -> None:
    create = AsyncMock(return_value=SimpleNamespace(choices=[]))
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(
        client=sdk_client,
        model="test-model",
    )

    with pytest.raises(LLMUnavailableError, match="empty response"):
        await client.chat([{"role": "user", "content": "你好"}])
