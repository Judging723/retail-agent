import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    ConversationServiceDependency,
    KnowledgeRetrieverDependency,
    LLMClientDependency,
)
from app.llm import LLMUnavailableError
from app.rag import RetrievedDocument
from app.schemas.conversation import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDetailResponse,
    KnowledgeSourceResponse,
    MessageResponse,
)
from app.services import ConversationClosedError, ConversationNotFoundError

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


@router.get(
    "/{conversation_no}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_no: str,
    service: ConversationServiceDependency,
) -> ConversationDetailResponse:
    try:
        conversation = await service.get_conversation(conversation_no)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from exc

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
    "/{conversation_no}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
)
async def chat(
    conversation_no: str,
    payload: ChatRequest,
    service: ConversationServiceDependency,
    llm_client: LLMClientDependency,
    knowledge_retriever: KnowledgeRetrieverDependency,
) -> ChatResponse:
    try:
        message, sources = await service.chat(
            conversation_no=conversation_no,
            user_content=payload.content,
            llm_client=llm_client,
            knowledge_retriever=knowledge_retriever,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from exc
    except ConversationClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation is closed",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        message_id=message.id,
        conversation_no=conversation_no,
        sequence_no=message.sequence_no,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        sources=_build_source_responses(sources),
    )


@router.post(
    "/{conversation_no}/chat/stream",
    response_class=StreamingResponse,
)
async def stream_chat(
    conversation_no: str,
    payload: ChatRequest,
    service: ConversationServiceDependency,
    llm_client: LLMClientDependency,
    knowledge_retriever: KnowledgeRetrieverDependency,
) -> StreamingResponse:
    try:
        prepared_chat = await service.prepare_chat(
            conversation_no=conversation_no,
            user_content=payload.content,
            knowledge_retriever=knowledge_retriever,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from exc
    except ConversationClosedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation is closed",
        ) from exc

    source_responses = _build_source_responses(prepared_chat.sources)

    async def event_stream() -> AsyncIterator[str]:
        yield _encode_sse_event(
            "sources",
            {"sources": [source.model_dump(mode="json") for source in source_responses]},
        )

        try:
            message = await service.complete_prepared_chat(
                prepared_chat=prepared_chat,
                user_content=payload.content,
                llm_client=llm_client,
            )
            yield _encode_sse_event("delta", {"content": message.content})
        except LLMUnavailableError as exc:
            yield _encode_sse_event("error", {"detail": str(exc)})
            return

        response = ChatResponse(
            message_id=message.id,
            conversation_no=conversation_no,
            sequence_no=message.sequence_no,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            sources=source_responses,
        )
        yield _encode_sse_event("done", response.model_dump(mode="json"))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _build_source_responses(
    sources: list[RetrievedDocument],
) -> list[KnowledgeSourceResponse]:
    return [
        KnowledgeSourceResponse(
            source_id=source.source_id,
            title=source.title,
            category=source.category,
            revision=source.revision,
            source_date=source.source_date,
            source_url=source.source_url,
            license=source.license,
        )
        for source in sources
    ]


def _encode_sse_event(event: str, data: object) -> str:
    serialized = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {serialized}\n\n"
