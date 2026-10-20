from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Message,
    MessageRole,
)
from app.repositories import (
    ConversationNotFoundError as RepositoryConversationNotFoundError,
)
from app.repositories import ConversationRepository, CustomerRepository
from app.services.exceptions import (
    ConversationClosedError,
    ConversationNotFoundError,
)


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._customers = CustomerRepository(session)
        self._conversations = ConversationRepository(session)

    async def create_conversation(
        self,
        *,
        customer_no: str,
        display_name: str | None,
    ) -> tuple[Conversation, Customer]:
        async with self._session.begin():
            customer = await self._customers.get_by_customer_no(customer_no)

            if customer is None:
                customer = await self._customers.create(
                    customer_no=customer_no,
                    display_name=display_name,
                )

            conversation = await self._conversations.create(
                conversation_no=self._generate_conversation_no(),
                customer_id=customer.id,
            )

            # 加载数据库生成的 created_at 和 updated_at。
            await self._session.refresh(conversation)

        return conversation, customer

    async def add_user_message(
        self,
        *,
        conversation_no: str,
        content: str,
    ) -> Message:
        async with self._session.begin():
            conversation = await self._conversations.get_by_conversation_no(conversation_no)

            if conversation is None:
                raise ConversationNotFoundError(conversation_no)

            if conversation.status == ConversationStatus.CLOSED:
                raise ConversationClosedError(conversation_no)

            try:
                message = await self._conversations.append_message(
                    conversation_id=conversation.id,
                    role=MessageRole.USER,
                    content=content,
                )
            except RepositoryConversationNotFoundError as exc:
                # 防止查询后、加锁前会话被其他事务删除。
                raise ConversationNotFoundError(conversation_no) from exc

            await self._session.refresh(message)

        return message

    async def get_conversation(
        self,
        conversation_no: str,
    ) -> Conversation:
        async with self._session.begin():
            conversation = await self._conversations.get_by_conversation_no(conversation_no)

            if conversation is None:
                raise ConversationNotFoundError(conversation_no)

        return conversation

    @staticmethod
    def _generate_conversation_no() -> str:
        suffix = uuid4().hex[:20].upper()
        return f"CONV-{suffix}"
