from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationStatus, MessageRole
from app.services import (
    ConversationClosedError,
    ConversationNotFoundError,
    ConversationService,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_conversation_lifecycle(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-SERVICE-{suffix}"

    service = ConversationService(db_session)

    conversation, customer = await service.create_conversation(
        customer_no=customer_no,
        display_name="Service 测试用户",
    )

    assert customer.id is not None
    assert customer.customer_no == customer_no
    assert conversation.id is not None
    assert conversation.conversation_no.startswith("CONV-")
    assert conversation.status == ConversationStatus.OPEN
    assert conversation.created_at is not None

    message = await service.add_user_message(
        conversation_no=conversation.conversation_no,
        content="请帮我查询物流状态",
    )

    assert message.id is not None
    assert message.sequence_no == 1
    assert message.role == MessageRole.USER
    assert message.created_at is not None

    loaded = await service.get_conversation(conversation.conversation_no)

    assert loaded.customer.customer_no == customer_no
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "请帮我查询物流状态"


async def test_get_missing_conversation(
    db_session: AsyncSession,
) -> None:
    service = ConversationService(db_session)

    with pytest.raises(
        ConversationNotFoundError,
        match="Conversation CONV-NOT-FOUND was not found",
    ):
        await service.get_conversation("CONV-NOT-FOUND")


async def test_closed_conversation_rejects_new_message(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    service = ConversationService(db_session)

    conversation, _ = await service.create_conversation(
        customer_no=f"CUST-CLOSED-{suffix}",
        display_name=None,
    )

    async with db_session.begin():
        conversation.status = ConversationStatus.CLOSED
        await db_session.flush()

    with pytest.raises(
        ConversationClosedError,
        match="is already closed",
    ):
        await service.add_user_message(
            conversation_no=conversation.conversation_no,
            content="关闭后不应该保存这条消息",
        )


async def test_existing_customer_can_create_multiple_conversations(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-SERVICE-{suffix}"

    service = ConversationService(db_session)
    conversation1, customer1 = await service.create_conversation(
        customer_no=customer_no,
        display_name="Service 测试用户",
    )
    conversation2, customer2 = await service.create_conversation(
        customer_no=customer_no,
        display_name="Service 测试用户",
    )
    assert customer1.id == customer2.id
    assert conversation1.id != conversation2.id
    assert conversation1.conversation_no != conversation2.conversation_no
