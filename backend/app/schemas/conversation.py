from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import ConversationStatus, MessageRole


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_no: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    display_name: str | None = Field(min_length=1, max_length=100, default=None)


class ConversationCreateResponse(BaseModel):
    conversation_no: str
    customer_no: str
    status: ConversationStatus
    created_at: datetime


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=4000)


class MessageResponse(BaseModel):
    message_id: int
    sequence_no: int
    role: MessageRole
    content: str
    created_at: datetime


class KnowledgeSourceResponse(BaseModel):
    source_id: str
    title: str


class ChatResponse(MessageResponse):
    conversation_no: str
    sources: list[KnowledgeSourceResponse]


class ConversationDetailResponse(BaseModel):
    conversation_no: str
    customer_no: str
    status: ConversationStatus
    summary: str | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]
