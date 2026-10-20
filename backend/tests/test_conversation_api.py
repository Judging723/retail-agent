from uuid import uuid4

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, ConversationStatus, Message

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_create_conversation(
    api_client: AsyncClient,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-API-{suffix}"

    response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": customer_no,
            "display_name": "API 测试用户",
        },
    )

    assert response.status_code == 201

    body = response.json()

    # 请你根据响应模型完成下面五个断言
    assert body["customer_no"] == customer_no
    assert body["conversation_no"].startswith("CONV-")
    assert body["status"] == "open"
    assert body["created_at"] is not None
    assert set(body) == {
        "customer_no",
        "conversation_no",
        "status",
        "created_at",
    }


async def test_create_conversation_rejects_invalid_customer_no(
    api_client: AsyncClient,
) -> None:
    response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": "CUST 001",
            "display_name": "无效用户",
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Input validation failed"
    assert any(detail["loc"] == ["body", "customer_no"] for detail in body["error"]["details"])


async def test_same_customer_can_create_multiple_conversations(
    api_client: AsyncClient,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-API-MULTI-{suffix}"
    payload = {
        "customer_no": customer_no,
        "display_name": "重复会话测试用户",
    }

    first_response = await api_client.post(
        "/api/v1/conversations",
        json=payload,
    )
    second_response = await api_client.post(
        "/api/v1/conversations",
        json=payload,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 201

    first_body = first_response.json()
    second_body = second_response.json()

    assert first_body["customer_no"] == customer_no
    assert second_body["customer_no"] == customer_no
    assert first_body["conversation_no"] != second_body["conversation_no"]


async def test_add_user_message(
    api_client: AsyncClient,
) -> None:
    suffix = uuid4().hex[:12]

    conversation_response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": f"CUST-MESSAGE-{suffix}",
            "display_name": "消息测试用户",
        },
    )
    assert conversation_response.status_code == 201

    conversation_no = conversation_response.json()["conversation_no"]

    response = await api_client.post(
        f"/api/v1/conversations/{conversation_no}/messages",
        json={
            "content": "  我的订单什么时候发货？  ",
        },
    )

    assert response.status_code == 201

    body = response.json()

    assert body["message_id"] > 0
    assert body["conversation_no"] == conversation_no
    assert body["sequence_no"] == 1
    assert body["role"] == "user"
    assert body["content"] == "我的订单什么时候发货？"
    assert body["created_at"] is not None


async def test_add_message_to_missing_conversation_returns_404(
    api_client: AsyncClient,
) -> None:
    conversation_no = "CONV-00000000000000000000"

    response = await api_client.post(
        f"/api/v1/conversations/{conversation_no}/messages",
        json={
            "content": "这条消息不应该被保存",
        },
    )

    assert response.status_code == 404

    body = response.json()

    assert body["error"]["code"] == "conversation_not_found"
    assert body["error"]["details"] == {
        "conversation_no": conversation_no,
    }
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]


async def test_add_message_to_closed_conversation_returns_409(
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]

    create_response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": f"CUST-CLOSED-API-{suffix}",
            "display_name": "关闭会话测试用户",
        },
    )
    assert create_response.status_code == 201

    conversation_no = create_response.json()["conversation_no"]

    async with db_session.begin():
        conversation = await db_session.scalar(
            select(Conversation).where(Conversation.conversation_no == conversation_no)
        )

        assert conversation is not None

        conversation.status = ConversationStatus.CLOSED
        conversation_id = conversation.id

    response = await api_client.post(
        f"/api/v1/conversations/{conversation_no}/messages",
        json={
            "content": "关闭后不应该保存这条消息",
        },
    )

    assert response.status_code == 409

    body = response.json()

    assert body["error"]["code"] == "conversation_closed"
    assert body["error"]["details"] == {
        "conversation_no": conversation_no,
    }

    message_count = await db_session.scalar(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    )

    assert message_count == 0


async def test_get_conversation_returns_ordered_messages(
    api_client: AsyncClient,
) -> None:
    suffix = uuid4().hex[:12]

    create_response = await api_client.post(
        "/api/v1/conversations",
        json={
            "customer_no": f"CUST-DETAIL-{suffix}",
            "display_name": "详情测试用户",
        },
    )
    assert create_response.status_code == 201

    conversation_no = create_response.json()["conversation_no"]

    contents = [
        "第一条消息",
        "第二条消息",
    ]

    for content in contents:
        message_response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/messages",
            json={"content": content},
        )
        assert message_response.status_code == 201

    response = await api_client.get(f"/api/v1/conversations/{conversation_no}")

    assert response.status_code == 200

    body = response.json()

    assert body["conversation_no"] == conversation_no
    assert body["customer_no"] == f"CUST-DETAIL-{suffix}"
    assert body["status"] == "open"
    assert body["summary"] is None
    assert body["created_at"] is not None
    assert body["updated_at"] is not None

    assert [message["sequence_no"] for message in body["messages"]] == [1, 2]

    assert [message["content"] for message in body["messages"]] == contents


async def test_get_missing_conversation_returns_404(
    api_client: AsyncClient,
) -> None:
    conversation_no = "CONV-00000000000000000000"

    response = await api_client.get(f"/api/v1/conversations/{conversation_no}")

    assert response.status_code == 404

    body = response.json()

    assert body["error"]["code"] == "conversation_not_found"
    assert body["error"]["details"] == {
        "conversation_no": conversation_no,
    }
    assert response.headers["X-Request-ID"] == body["error"]["request_id"]
