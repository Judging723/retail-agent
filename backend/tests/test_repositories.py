from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MessageRole
from app.repositories import (
    ConversationNotFoundError,
    ConversationRepository,
    CustomerRepository,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_create_and_load_conversation(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-TEST-{suffix}"
    conversation_no = f"CONV-TEST-{suffix}"

    customers = CustomerRepository(db_session)
    conversations = ConversationRepository(db_session)

    customer = await customers.create(
        customer_no=customer_no,
        display_name="集成测试用户",
    )

    found_customer = await customers.get_by_customer_no(customer_no)

    assert found_customer is not None
    assert found_customer.id == customer.id

    conversation = await conversations.create(
        conversation_no=conversation_no,
        customer_id=customer.id,
    )

    user_message = await conversations.append_message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content="我的订单什么时候发货？",
    )
    assistant_message = await conversations.append_message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content="请提供您的订单编号。",
    )

    assert user_message.sequence_no == 1
    assert assistant_message.sequence_no == 2

    # 清除当前 Session 中的缓存状态，确保下面的数据从数据库重载。
    db_session.expire_all()

    loaded = await conversations.get_by_conversation_no(conversation_no)

    assert loaded is not None
    assert loaded.customer.customer_no == customer_no
    assert [message.sequence_no for message in loaded.messages] == [1, 2]
    assert [message.role for message in loaded.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [message.content for message in loaded.messages] == [
        "我的订单什么时候发货？",
        "请提供您的订单编号。",
    ]


async def test_append_message_rejects_missing_conversation(
    db_session: AsyncSession,
) -> None:
    conversations = ConversationRepository(db_session)

    with pytest.raises(
        ConversationNotFoundError,
        match="Conversation with id -1 was not found",
    ):
        await conversations.append_message(
            conversation_id=-1,
            role=MessageRole.USER,
            content="这条消息不应该被保存",
        )
