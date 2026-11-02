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


async def test_openai_compatible_client_returns_tool_call() -> None:
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="get_order_status",
                                    arguments='{"order_no":"ORDER-001"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )
    )
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(client=sdk_client, model="test-model")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "parameters": {"type": "object"},
            },
        }
    ]

    turn = await client.complete_with_tools(
        [{"role": "user", "content": "查订单 ORDER-001"}],
        tools,
    )

    assert turn.content is None
    assert turn.tool_calls[0].id == "call-1"
    assert turn.tool_calls[0].name == "get_order_status"
    assert turn.tool_calls[0].arguments == '{"order_no":"ORDER-001"}'
    create.assert_awaited_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "查订单 ORDER-001"}],
        tools=tools,
        tool_choice="auto",
    )


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


async def test_openai_compatible_client_streams_text_with_tools() -> None:
    async def completion_stream():
        for content in ["正在", "查询", "。"]:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))]
            )

    create = AsyncMock(return_value=completion_stream())
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(client=sdk_client, model="test-model")
    tools = [{"type": "function", "function": {"name": "get_order_status"}}]

    events = [
        event
        async for event in client.stream_with_tools(
            [{"role": "user", "content": "查询订单"}],
            tools,
        )
    ]

    assert [event.content_delta for event in events[:-1]] == ["正在", "查询", "。"]
    assert events[-1].completed_turn is not None
    assert events[-1].completed_turn.content == "正在查询。"
    assert events[-1].completed_turn.tool_calls == ()
    create.assert_awaited_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "查询订单"}],
        tools=tools,
        tool_choice="auto",
        stream=True,
    )


async def test_openai_compatible_client_reconstructs_streamed_tool_call() -> None:
    async def completion_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(
                                    name="get_order_",
                                    arguments='{"order_no":"',
                                ),
                            )
                        ],
                    )
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name="status",
                                    arguments='ORDER-001"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    create = AsyncMock(return_value=completion_stream())
    sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAICompatibleLLMClient(client=sdk_client, model="test-model")

    events = [
        event
        async for event in client.stream_with_tools(
            [{"role": "user", "content": "查 ORDER-001"}],
            [{"type": "function", "function": {"name": "get_order_status"}}],
        )
    ]

    assert len(events) == 1
    turn = events[0].completed_turn
    assert turn is not None
    assert turn.content is None
    assert turn.tool_calls[0].id == "call-1"
    assert turn.tool_calls[0].name == "get_order_status"
    assert turn.tool_calls[0].arguments == '{"order_no":"ORDER-001"}'
