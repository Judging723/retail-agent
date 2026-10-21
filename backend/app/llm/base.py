from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    model: str
    finish_reason: str | None = None


class LLMClient(Protocol):
    async def chat(
        self,
        *,
        messages: Sequence[ChatMessage],
    ) -> ChatResponse: ...
