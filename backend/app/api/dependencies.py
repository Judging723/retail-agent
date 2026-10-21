from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.services.conversation import ConversationService
from app.agents import CustomerSupportAgent
from app.llm import LLMClient, LLMUnavailableError

DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_conversation_service(
    session: DatabaseSession,
) -> ConversationService:
    # 使用 session 创建并返回 ConversationService
    return ConversationService(session)


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]


def get_llm_client() -> LLMClient:
    raise LLMUnavailableError()


LLMClientDependency = Annotated[
    LLMClient,
    Depends(get_llm_client),
]


def get_customer_support_agent(
    conversation_service: ConversationServiceDependency,
    llm_client: LLMClientDependency,
) -> CustomerSupportAgent:
    return CustomerSupportAgent(
        conversation_service=conversation_service,
        llm_client=llm_client,
    )


CustomerSupportAgentDependency = Annotated[
    CustomerSupportAgent,
    Depends(get_customer_support_agent),
]
