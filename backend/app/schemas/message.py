from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import MessageRole


class MessageCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(
        min_length=1,
        max_length=4000,
    )


class MessageResponse(BaseModel):
    message_id: int
    sequence_no: int
    role: MessageRole
    content: str
    created_at: datetime


class MessageCreateResponse(MessageResponse):
    conversation_no: str
