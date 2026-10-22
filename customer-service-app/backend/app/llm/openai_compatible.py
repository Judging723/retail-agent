from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

ChatMessage = dict[str, str]


class LLMUnavailableError(RuntimeError):
    """The configured model service cannot return a usable answer."""


class LLMClient(Protocol):
    async def chat(self, messages: Sequence[ChatMessage]) -> str: ...

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
