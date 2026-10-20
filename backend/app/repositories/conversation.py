from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
)
from app.repositories.exceptions import ConversationNotFoundError


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        conversation_no: str,
        customer_id: int,
        status: ConversationStatus = ConversationStatus.OPEN,
    ) -> Conversation:
        conversation = Conversation(
            conversation_no=conversation_no,
            customer_id=customer_id,
            status=status,
        )
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def get_by_conversation_no(
        self,
        conversation_no: str,
    ) -> Conversation | None:
        statement = (
            select(Conversation)
            .where(Conversation.conversation_no == conversation_no)
            .options(
                selectinload(Conversation.customer),
                selectinload(Conversation.messages),
            )
            .execution_options(populate_existing=True)
        )
        return await self._session.scalar(statement)

    async def append_message(
        self,
        *,
        conversation_id: int,
        role: MessageRole,
        content: str,
    ) -> Message:
        locked_conversation_id = await self._session.scalar(
            select(Conversation.id).where(Conversation.id == conversation_id).with_for_update()
        )

        if locked_conversation_id is None:
            raise ConversationNotFoundError(conversation_id)

        next_sequence_no = await self._session.scalar(
            select(func.coalesce(func.max(Message.sequence_no), 0) + 1).where(
                Message.conversation_id == conversation_id
            )
        )

        if next_sequence_no is None:
            raise RuntimeError("Failed to calculate the next message sequence")

        message = Message(
            conversation_id=conversation_id,
            sequence_no=int(next_sequence_no),
            role=role,
            content=content,
        )
        self._session.add(message)
        await self._session.flush()
        return message
