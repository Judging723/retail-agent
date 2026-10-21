from app.llm import (
    ChatMessage,
    ChatRole,
    LLMClient,
)
from app.models import Message, MessageRole
from app.services import ConversationService

DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业、谨慎的电商客服助手。"
    "只根据会话和可用工具提供的信息回答。"
    "信息不足时先追问，不得编造订单、物流、退款或赔付状态。"
)

_MESSAGE_ROLE_MAP = {
    MessageRole.SYSTEM: ChatRole.SYSTEM,
    MessageRole.USER: ChatRole.USER,
    MessageRole.ASSISTANT: ChatRole.ASSISTANT,
}


class CustomerSupportAgent:
    def __init__(
        self,
        *,
        conversation_service: ConversationService,
        llm_client: LLMClient,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._conversation_service = conversation_service
        self._llm_client = llm_client
        self._system_prompt = system_prompt

    async def respond(
        self,
        *,
        conversation_no: str,
        user_content: str,
    ) -> Message:
        await self._conversation_service.add_user_message(
            conversation_no=conversation_no,
            content=user_content,
        )

        conversation = await self._conversation_service.get_conversation(conversation_no)

        chat_messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=self._system_prompt,
            ),
            *[self._to_chat_message(message) for message in conversation.messages],
        ]

        llm_response = await self._llm_client.chat(messages=chat_messages)

        return await self._conversation_service.add_assistant_message(
            conversation_no=conversation_no,
            content=llm_response.content,
        )

    @staticmethod
    def _to_chat_message(message: Message) -> ChatMessage:
        role = _MESSAGE_ROLE_MAP.get(message.role)

        if role is None:
            raise ValueError(f"Unsupported message role: {message.role}")

        return ChatMessage(
            role=role,
            content=message.content,
        )
