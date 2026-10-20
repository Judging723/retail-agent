from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.services.conversation import ConversationService

DatabaseSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_conversation_service(
    session: DatabaseSession,
) -> ConversationService:
    # 使用 session 创建并返回 ConversationService
    return ConversationService(session)


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]
