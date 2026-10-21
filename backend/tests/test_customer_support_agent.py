from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import CustomerSupportAgent
from app.llm import (
    ChatResponse,
    ChatRole,
    FakeLLMClient,
)
from app.models import MessageRole
from app.services import ConversationService

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_agent_builds_prompt_and_persists_reply(
    db_session: AsyncSession,
) -> None:
    service = ConversationService(db_session)

    conversation, _ = await service.create_conversation(
        customer_no=f"CUST-AGENT-{uuid4().hex[:12]}",
        display_name="Agent 测试用户",
    )

    fake_llm = FakeLLMClient(
        ChatResponse(
            content="请提供您的订单编号。",
            model="fake-model",
            finish_reason="stop",
        )
    )
    agent = CustomerSupportAgent(
        conversation_service=service,
        llm_client=fake_llm,
    )

    assistant_message = await agent.respond(
        conversation_no=conversation.conversation_no,
        user_content="帮我查询物流。",
    )

    assert assistant_message.sequence_no == 2
    assert assistant_message.role == MessageRole.ASSISTANT
    assert assistant_message.content == "请提供您的订单编号。"

    assert len(fake_llm.calls) == 1

    prompt = fake_llm.calls[0]

    assert [message.role for message in prompt] == [
        ChatRole.SYSTEM,
        ChatRole.USER,
    ]
    assert "不得编造" in prompt[0].content
    assert prompt[1].content == "帮我查询物流。"

    loaded = await service.get_conversation(conversation.conversation_no)

    assert [message.sequence_no for message in loaded.messages] == [1, 2]

    assert [message.role for message in loaded.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]

    assert [message.content for message in loaded.messages] == [
        "帮我查询物流。",
        "请提供您的订单编号。",
    ]
