from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

ChatMessage = dict[str, Any]
ToolDefinition = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FunctionToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    content: str | None
    tool_calls: tuple[FunctionToolCall, ...]


class LLMUnavailableError(RuntimeError):
    """The configured model service cannot return a usable answer."""


class LLMClient(Protocol):
    async def chat(self, messages: Sequence[ChatMessage]) -> str: ...

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    async def chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
            )
        except (
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as exc:
            raise LLMUnavailableError("Model service is unavailable") from exc

        if not completion.choices:
            raise LLMUnavailableError("Model service returned an empty response")

        choice = completion.choices[0]
        content = choice.message.content

        if not isinstance(content, str) or not content.strip():
            raise LLMUnavailableError("Model service returned an empty response")

        return content.strip()

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                tools=list(tools),
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )
        except (
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as exc:
            raise LLMUnavailableError("Model service is unavailable") from exc

        if not completion.choices:
            raise LLMUnavailableError("Model service returned an empty response")

        message = completion.choices[0].message
        content = message.content.strip() if isinstance(message.content, str) else None
        tool_calls = tuple(
            FunctionToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (message.tool_calls or [])
        )

        if not content and not tool_calls:
            raise LLMUnavailableError("Model service returned an empty response")

        return AssistantTurn(
            content=content or None,
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[str]:
        try:
            completion_stream = await self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                stream=True,
            )

            has_text = False
            async for chunk in completion_stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content
                if not isinstance(content, str) or not content:
                    continue

                has_text = has_text or bool(content.strip())
                yield content
        except (
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as exc:
            raise LLMUnavailableError("Model service is unavailable") from exc

        if not has_text:
            raise LLMUnavailableError("Model service returned an empty response")
