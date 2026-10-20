from fastapi import APIRouter, status

from app.api.dependencies import ConversationServiceDependency
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDetailResponse,
)
from app.schemas.error import ErrorResponse
from app.schemas.message import (
    MessageCreateRequest,
    MessageCreateResponse,
    MessageResponse,
)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)


@router.post(
    "",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Request validation failed",
        },
    },
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


@router.get(
    "/{conversation_no}",
    response_model=ConversationDetailResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Conversation not found",
        },
    },
)
async def get_conversation(
    conversation_no: str,
    service: ConversationServiceDependency,
) -> ConversationDetailResponse:
    conversation = await service.get_conversation(conversation_no)

    return ConversationDetailResponse(
        conversation_no=conversation.conversation_no,
        customer_no=conversation.customer.customer_no,
        status=conversation.status,
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageResponse(
                message_id=message.id,
                sequence_no=message.sequence_no,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in conversation.messages
        ],
    )


@router.post(
    "/{conversation_no}/messages",
    response_model=MessageCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Conversation not found",
        },
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "Conversation is closed",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "Request validation failed",
        },
    },
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
