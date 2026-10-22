from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db_session
from app.llm import LLMClient, OpenAICompatibleLLMClient
from app.rag import CrossEncoderReranker, KnowledgeRetriever
from app.services.conversation import ConversationService

DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_conversation_service(
    session: DatabaseSession,
) -> ConversationService:
    # 使用 session 创建并返回 ConversationService
    return ConversationService(session)


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]


@lru_cache
def get_knowledge_retriever() -> KnowledgeRetriever:
    settings = get_settings()
    reranker = CrossEncoderReranker() if settings.rag_reranker_enabled else None
    return KnowledgeRetriever.from_milvus(
        settings.milvus_uri,
        collection_name=settings.milvus_collection,
        min_score=settings.rag_min_score,
        reranker=reranker,
        rerank_candidates=settings.rag_reranker_candidates,
    )


KnowledgeRetrieverDependency = Annotated[
    KnowledgeRetriever,
    Depends(get_knowledge_retriever),
]


async def get_llm_client() -> AsyncIterator[LLMClient]:
    settings = get_settings()
    api_key = settings.llm_api_key.get_secret_value()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM API key is not configured",
        )

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=settings.llm_api_base,
    ) as sdk_client:
        yield OpenAICompatibleLLMClient(
            client=sdk_client,
            model=settings.llm_model,
        )


LLMClientDependency = Annotated[
    LLMClient,
    Depends(get_llm_client),
]
