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


async def test_openai_compatible_client_streams_text() -> None:
    async def completion_stream():
        for content in ["您", "好", "！"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    create = AsyncMock(return_value=completion_stream())
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(
        client=sdk_client,
        model="test-model",
    )

    chunks = [chunk async for chunk in client.stream([{"role": "user", "content": "你好"}])]

    assert chunks == ["您", "好", "！"]
    create.assert_awaited_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
    )


async def test_openai_compatible_client_rejects_empty_stream() -> None:
    async def empty_stream():
        yield SimpleNamespace(choices=[])

    create = AsyncMock(return_value=empty_stream())
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(
        client=sdk_client,
        model="test-model",
    )

    with pytest.raises(LLMUnavailableError, match="empty response"):
        _ = [chunk async for chunk in client.stream([{"role": "user", "content": "你好"}])]
