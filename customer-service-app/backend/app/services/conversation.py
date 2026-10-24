from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents import CustomerSupportAgent, OrderToolExecutor
from app.llm import ChatMessage, LLMClient
from app.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Message,
    MessageRole,
)
from app.rag import KnowledgeRetriever, RetrievedDocument
from app.services.policy_guardrail import policy_fallback

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业、谨慎的电商客服助手。"
    "请根据会话内容回答；信息不足时先追问，"
    "不得编造订单、物流、退款或赔付状态。"
    "涉及具体订单状态时，必须使用工具查询；订单号不足时先向用户追问。"
    "检索到的公开数据是购物澄清案例，不是平台业务规则；"
    "只能借鉴其中的澄清思路，不能把案例内容说成当前用户或平台的事实。"
)


class ConversationNotFoundError(Exception):
    pass


class ConversationClosedError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PreparedChat:
    conversation_id: int
    customer_id: int
    next_sequence: int
    prompt: list[ChatMessage]
    sources: list[RetrievedDocument]
    deterministic_response: str | None


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
        knowledge_retriever: KnowledgeRetriever,
        idempotency_key: str | None = None,
    ) -> tuple[Message, list[RetrievedDocument]]:
        prepared_chat = await self.prepare_chat(
            conversation_no=conversation_no,
            user_content=user_content,
            knowledge_retriever=knowledge_retriever,
        )

        assistant_message = await self.complete_prepared_chat(
            prepared_chat=prepared_chat,
            user_content=user_content,
            llm_client=llm_client,
            idempotency_key=idempotency_key,
        )

        return assistant_message, prepared_chat.sources

    async def complete_prepared_chat(
        self,
        *,
        prepared_chat: PreparedChat,
        user_content: str,
        llm_client: LLMClient,
        idempotency_key: str | None = None,
    ) -> Message:
        assistant_content = prepared_chat.deterministic_response
        if assistant_content is None:
            agent = CustomerSupportAgent(
                llm_client=llm_client,
                order_tool_executor=OrderToolExecutor(self._session),
            )
            assistant_content = await agent.respond(
                prepared_chat.prompt,
                conversation_id=prepared_chat.conversation_id,
                customer_id=prepared_chat.customer_id,
                idempotency_key=idempotency_key
                or (
                    f"conversation:{prepared_chat.conversation_id}:"
                    f"sequence:{prepared_chat.next_sequence}"
                ),
            )
        assistant_message = await self.save_chat_exchange(
            conversation_id=prepared_chat.conversation_id,
            next_sequence=prepared_chat.next_sequence,
            user_content=user_content,
            assistant_content=assistant_content,
        )

        return assistant_message

    async def prepare_chat(
        self,
        *,
        conversation_no: str,
        user_content: str,
        knowledge_retriever: KnowledgeRetriever,
    ) -> PreparedChat:
        async with self._session.begin():
            conversation = await self._find_conversation(conversation_no)
            self._ensure_available(conversation, conversation_no)

            next_sequence = (
                conversation.messages[-1].sequence_no + 1 if conversation.messages else 1
            )
            deterministic_response = policy_fallback(user_content)
            retrieved_documents = (
                []
                if deterministic_response is not None
                else knowledge_retriever.search(user_content)
            )
            system_messages: list[ChatMessage] = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
            ]
            if retrieved_documents:
                system_messages.append(
                    {
                        "role": "system",
                        "content": self._build_knowledge_context(retrieved_documents),
                    }
                )

            prompt: list[ChatMessage] = [
                *system_messages,
                *[
                    {"role": message.role.value, "content": message.content}
                    for message in conversation.messages
                ],
                {"role": "user", "content": user_content},
            ]

        return PreparedChat(
            conversation_id=conversation.id,
            customer_id=conversation.customer_id,
            next_sequence=next_sequence,
            prompt=prompt,
            sources=retrieved_documents,
            deterministic_response=deterministic_response,
        )

    async def save_chat_exchange(
        self,
        *,
        conversation_id: int,
        next_sequence: int,
        user_content: str,
        assistant_content: str,
    ) -> Message:
        async with self._session.begin():
            user_message = Message(
                conversation_id=conversation_id,
                sequence_no=next_sequence,
                role=MessageRole.USER,
                content=user_content,
            )
            assistant_message = Message(
                conversation_id=conversation_id,
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

    @staticmethod
    def _build_knowledge_context(documents: list[RetrievedDocument]) -> str:
        sections = [
            (
                f"[知识{i}] {document.title}"
                f"（数据版本 {document.revision}，对话日期 {document.source_date}，"
                f"许可 {document.license}）\n"
                f"{document.content}"
            )
            for i, document in enumerate(documents, start=1)
        ]
        return (
            "以下内容来自公开的真实购物澄清案例，仅用于帮助识别歧义并设计追问。"
            "它们不是退款、物流或售后政策，也不是当前用户的历史记录。"
            "请借鉴澄清方式，不要照搬案例事实。\n\n" + "\n\n".join(sections)
        )
