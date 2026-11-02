import json
import logging
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    ConversationServiceDependency,
    CurrentCustomerDependency,
    KnowledgeRetrieverDependency,
    LLMClientDependency,
)
from app.llm import LLMUnavailableError
from app.rag import RetrievedDocument
from app.schemas.conversation import (
    ChatRequest,
    ChatResponse,
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
stream_logger = logging.getLogger("app.chat_stream")


@router.post(
    "",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    service: ConversationServiceDependency,
    customer: CurrentCustomerDependency,
) -> ConversationCreateResponse:
    conversation = await service.create_conversation(customer_id=customer.id)
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
    customer: CurrentCustomerDependency,
) -> ConversationDetailResponse:
    try:
        conversation = await service.get_conversation(conversation_no, customer.id)
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
    customer: CurrentCustomerDependency,
    llm_client: LLMClientDependency,
    knowledge_retriever: KnowledgeRetrieverDependency,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
            pattern=r"^[A-Za-z0-9._:-]+$",
        ),
    ] = None,
) -> ChatResponse:
    try:
        message, sources = await service.chat(
            conversation_no=conversation_no,
            customer_id=customer.id,
            user_content=payload.content,
            llm_client=llm_client,
            knowledge_retriever=knowledge_retriever,
            idempotency_key=idempotency_key,
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
    request: Request,
    conversation_no: str,
    payload: ChatRequest,
    service: ConversationServiceDependency,
    customer: CurrentCustomerDependency,
    llm_client: LLMClientDependency,
    knowledge_retriever: KnowledgeRetrieverDependency,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=8,
            max_length=128,
            pattern=r"^[A-Za-z0-9._:-]+$",
        ),
    ] = None,
) -> StreamingResponse:
    stream_started_at = request.state.started_at
    try:
        prepared_chat = await service.prepare_chat(
            conversation_no=conversation_no,
            customer_id=customer.id,
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
        first_delta_ms: float | None = None
        delta_count = 0
        character_count = 0
        stream_status = "interrupted"
        try:
            yield _encode_sse_event(
                "sources",
                {"sources": [source.model_dump(mode="json") for source in source_responses]},
            )

            try:
                message = None
                async for stream_event in service.stream_prepared_chat(
                    prepared_chat=prepared_chat,
                    user_content=payload.content,
                    llm_client=llm_client,
                    idempotency_key=idempotency_key,
                ):
                    if stream_event.content_delta is not None:
                        if first_delta_ms is None:
                            first_delta_ms = (perf_counter() - stream_started_at) * 1000
                        delta_count += 1
                        character_count += len(stream_event.content_delta)
                        yield _encode_sse_event(
                            "delta",
                            {"content": stream_event.content_delta},
                        )
                    if stream_event.completed_message is not None:
                        message = stream_event.completed_message
            except LLMUnavailableError as exc:
                stream_status = "model_error"
                yield _encode_sse_event("error", {"detail": str(exc)})
                return

            if message is None:
                stream_status = "incomplete"
                yield _encode_sse_event(
                    "error",
                    {"detail": "Chat stream ended before the message was saved"},
                )
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
            stream_status = "completed"
            yield _encode_sse_event("done", response.model_dump(mode="json"))
        finally:
            total_ms = (perf_counter() - stream_started_at) * 1000
            stream_logger.info(
                "chat_stream_finished request_id=%s conversation_no=%s status=%s "
                "ttft_ms=%.2f total_ms=%.2f delta_count=%d character_count=%d",
                request.state.request_id,
                conversation_no,
                stream_status,
                first_delta_ms if first_delta_ms is not None else -1,
                total_ms,
                delta_count,
                character_count,
            )

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
