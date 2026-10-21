from collections.abc import Sequence

from app.llm.base import (
    ChatMessage,
    ChatResponse,
)


class FakeLLMClient:
    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.calls: list[tuple[ChatMessage, ...]] = []

    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
    ) -> ChatResponse:
        self.calls.append(tuple(messages))
        return self._response
