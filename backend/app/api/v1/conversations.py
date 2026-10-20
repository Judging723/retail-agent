from fastapi import APIRouter, status

from app.api.dependencies import ConversationServiceDependency
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationCreateResponse,
)
from app.schemas.message import (
    MessageCreateRequest,
    MessageCreateResponse,
)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.post(
    "",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreateRequest,
    service: ConversationServiceDependency,
) -> ConversationCreateResponse:
    conversation, customer = await service.create_conversation(
        customer_no=payload.customer_no,
        display_name=payload.display_name,
    )
    return ConversationCreateResponse(
        conversation_no=conversation.conversation_no,
        customer_no=customer.customer_no,
        status=conversation.status,
        created_at=conversation.created_at,
    )


@router.post(
    "/{conversation_no}/messages",
    response_model=MessageCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_user_message(
    conversation_no: str,
    payload: MessageCreateRequest,
    service: ConversationServiceDependency,
) -> MessageCreateResponse:
    message = await service.add_user_message(
        conversation_no=conversation_no,
        content=payload.content,
    )

    return MessageCreateResponse(
        message_id=message.id,
        conversation_no=conversation_no,
        sequence_no=message.sequence_no,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )
