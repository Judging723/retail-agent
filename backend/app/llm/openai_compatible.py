from collections.abc import Sequence
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
)

from app.llm.base import (
    ChatMessage,
    ChatResponse,
)
from app.llm.exceptions import (
    LLMInvalidResponseError,
    LLMUnavailableError,
)


class _CompletionsResource(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _ChatResource(Protocol):
    completions: _CompletionsResource


class _AsyncChatClient(Protocol):
    chat: _ChatResource


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        client: _AsyncChatClient,
        model: str,
    ) -> None:
        self._client = client
        self._model = model

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
    ) -> ChatResponse:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": message.role.value,
                        "content": message.content,
                    }
                    for message in messages
                ],
            )
        except (
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as exc:
            raise LLMUnavailableError() from exc

        if not completion.choices:
            raise LLMInvalidResponseError()

        choice = completion.choices[0]
        content = choice.message.content

        if not isinstance(content, str) or not content.strip():
            raise LLMInvalidResponseError()

        return ChatResponse(
            content=content.strip(),
            model=completion.model,
            finish_reason=choice.finish_reason,
        )
