from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm import ChatMessage, LLMClient
from app.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Message,
    MessageRole,
)

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业、谨慎的电商客服助手。"
    "请根据会话内容回答；信息不足时先追问，"
    "不得编造订单、物流、退款或赔付状态。"
)


class ConversationNotFoundError(Exception):
    pass


class ConversationClosedError(Exception):
    pass


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_conversation(
        self,
        *,
        customer_no: str,
        display_name: str | None,
    ) -> tuple[Conversation, Customer]:
        async with self._session.begin():
            customer = await self._session.scalar(
                select(Customer).where(Customer.customer_no == customer_no)
            )

            if customer is None:
                customer = Customer(
                    customer_no=customer_no,
                    display_name=display_name,
                )
                self._session.add(customer)
                await self._session.flush()

            conversation = Conversation(
                conversation_no=self._generate_conversation_no(),
                customer_id=customer.id,
            )
            self._session.add(conversation)
            await self._session.flush()

            await self._session.refresh(conversation)

        return conversation, customer

    async def chat(
        self,
        *,
        conversation_no: str,
        user_content: str,
        llm_client: LLMClient,
    ) -> Message:
        async with self._session.begin():
            conversation = await self._find_conversation(conversation_no)
            self._ensure_available(conversation, conversation_no)

            next_sequence = (
                conversation.messages[-1].sequence_no + 1 if conversation.messages else 1
            )
            prompt: list[ChatMessage] = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                *[
                    {"role": message.role.value, "content": message.content}
                    for message in conversation.messages
                ],
                {"role": "user", "content": user_content},
            ]

            assistant_content = await llm_client.chat(prompt)

            user_message = Message(
                conversation_id=conversation.id,
                sequence_no=next_sequence,
                role=MessageRole.USER,
                content=user_content,
            )
            assistant_message = Message(
                conversation_id=conversation.id,
                sequence_no=next_sequence + 1,
                role=MessageRole.ASSISTANT,
                content=assistant_content,
            )
            self._session.add_all([user_message, assistant_message])
            await self._session.flush()
            await self._session.refresh(assistant_message)

        return assistant_message

    async def get_conversation(
        self,
        conversation_no: str,
    ) -> Conversation:
        async with self._session.begin():
            conversation = await self._find_conversation(conversation_no)

            if conversation is None:
                raise ConversationNotFoundError

        return conversation

    async def _find_conversation(self, conversation_no: str) -> Conversation | None:
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

    @staticmethod
    def _ensure_available(
        conversation: Conversation | None,
        conversation_no: str,
    ) -> None:
        if conversation is None:
            raise ConversationNotFoundError

        if conversation.status == ConversationStatus.CLOSED:
            raise ConversationClosedError(conversation_no)

    @staticmethod
    def _generate_conversation_no() -> str:
        suffix = uuid4().hex[:20].upper()
        return f"CONV-{suffix}"
