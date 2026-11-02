from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.security import decode_access_token
from app.llm import LLMClient, OpenAICompatibleLLMClient
from app.models import Customer
from app.rag import CrossEncoderReranker, KnowledgeRetriever, KnowledgeRuntime, KnowledgeSearcher
from app.services.auth import AuthService
from app.services.conversation import ConversationService

DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]
bearer_scheme = HTTPBearer(auto_error=False)


def get_auth_service(session: DatabaseSession) -> AuthService:
    return AuthService(session)


AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]


async def get_current_customer(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    session: DatabaseSession,
) -> Customer:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing access token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized

    try:
        claims = decode_access_token(credentials.credentials, get_settings())
        customer_id = int(str(claims["sub"]))
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise unauthorized from exc

    async with session.begin():
        customer = await session.scalar(select(Customer).where(Customer.id == customer_id))

    if customer is None:
        raise unauthorized

    return customer


CurrentCustomerDependency = Annotated[Customer, Depends(get_current_customer)]


def get_conversation_service(
    session: DatabaseSession,
) -> ConversationService:
    # 使用 session 创建并返回 ConversationService
    return ConversationService(session)


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]


def _build_knowledge_retriever() -> KnowledgeRetriever:
    settings = get_settings()
    reranker = CrossEncoderReranker() if settings.rag_reranker_enabled else None
    return KnowledgeRetriever.from_milvus(
        settings.milvus_uri,
        collection_name=settings.milvus_collection,
        min_score=settings.rag_min_score,
        reranker=reranker,
        rerank_candidates=settings.rag_reranker_candidates,
    )


@lru_cache
def get_knowledge_runtime() -> KnowledgeRuntime:
    return KnowledgeRuntime(_build_knowledge_retriever)


def get_knowledge_retriever() -> KnowledgeSearcher:
    return get_knowledge_runtime()


KnowledgeRetrieverDependency = Annotated[
    KnowledgeSearcher,
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
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    ) as sdk_client:
        yield OpenAICompatibleLLMClient(
            client=sdk_client,
            model=settings.llm_model,
        )


LLMClientDependency = Annotated[
    LLMClient,
    Depends(get_llm_client),
]
