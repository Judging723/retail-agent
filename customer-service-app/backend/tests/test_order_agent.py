import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import OrderToolExecutor
from app.api.dependencies import get_llm_client
from app.llm import AssistantTurn, ChatMessage, FunctionToolCall, ToolDefinition
from app.models import (
    Conversation,
    Customer,
    Order,
    OrderStatus,
    Shipment,
    ShipmentStatus,
    ToolAuditLog,
    ToolAuditStatus,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


def parse_sse_events(content: str) -> list[tuple[str, dict]]:
    events = []
    for block in content.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


class ScriptedOrderLLM:
    def __init__(self, order_no: str) -> None:
        self.order_no = order_no
        self.tool_result: dict | None = None

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("Agent should use complete_with_tools")

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        if messages[-1]["role"] != "tool":
            assert tools[0]["function"]["name"] == "get_order_status"
            return AssistantTurn(
                content=None,
                tool_calls=(
                    FunctionToolCall(
                        id="call-order-1",
                        name="get_order_status",
                        arguments=json.dumps({"order_no": self.order_no}),
                    ),
                ),
            )

        self.tool_result = json.loads(messages[-1]["content"])
        if self.tool_result["ok"]:
            order = self.tool_result["order"]
            shipment = order["shipment"]
            return AssistantTurn(
                content=(
                    f"订单 {order['order_no']} 当前为 {order['status']}，"
                    f"物流状态为 {shipment['status']}。"
                ),
                tool_calls=(),
            )
        return AssistantTurn(
            content="当前账户下未找到该订单，请核对订单号。",
            tool_calls=(),
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if not messages:
            yield ""
        raise AssertionError("This test does not use streaming")


async def test_agent_queries_current_customers_order_and_audits(
    application: FastAPI,
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-AGENT-{suffix}"
    order_no = f"ORDER-{suffix}"
    create_response = await api_client.post(
        "/api/v1/conversations",
        json={"customer_no": customer_no, "display_name": "Agent 测试用户"},
    )
    conversation_no = create_response.json()["conversation_no"]

    async with db_session.begin():
        customer = await db_session.scalar(
            select(Customer).where(Customer.customer_no == customer_no)
        )
        assert customer is not None
        order = Order(
            order_no=order_no,
            customer_id=customer.id,
            status=OrderStatus.SHIPPED,
            total_amount=Decimal("199.00"),
            currency="CNY",
        )
        order.shipment = Shipment(
            carrier="顺丰速运",
            tracking_no=f"SF{suffix}",
            status=ShipmentStatus.IN_TRANSIT,
            latest_event="快件已到达上海转运中心",
        )
        db_session.add(order)

    fake_llm = ScriptedOrderLLM(order_no)
    application.dependency_overrides[get_llm_client] = lambda: fake_llm
    try:
        response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat/stream",
            json={"content": f"帮我查一下订单 {order_no}"},
        )
    finally:
        application.dependency_overrides.pop(get_llm_client, None)

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert [event for event, _ in events] == ["sources", "delta", "done"]
    assert events[-1][1]["content"] == (f"订单 {order_no} 当前为 shipped，物流状态为 in_transit。")
    assert fake_llm.tool_result is not None
    assert fake_llm.tool_result["order"]["shipment"]["carrier"] == "顺丰速运"

    async with db_session.begin():
        audit = await db_session.scalar(
            select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "call-order-1")
        )
    assert audit is not None
    assert audit.status == ToolAuditStatus.SUCCESS
    assert audit.arguments == {"order_no": order_no}
    assert audit.error_code is None


async def test_agent_cannot_query_another_customers_order(
    application: FastAPI,
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    own_customer_no = f"CUST-OWN-{suffix}"
    foreign_order_no = f"ORDER-FOREIGN-{suffix}"
    create_response = await api_client.post(
        "/api/v1/conversations",
        json={"customer_no": own_customer_no, "display_name": "当前客户"},
    )
    conversation_no = create_response.json()["conversation_no"]

    async with db_session.begin():
        foreign_customer = Customer(
            customer_no=f"CUST-FOREIGN-{suffix}",
            display_name="其他客户",
        )
        db_session.add(foreign_customer)
        await db_session.flush()
        db_session.add(
            Order(
                order_no=foreign_order_no,
                customer_id=foreign_customer.id,
                status=OrderStatus.PAID,
                total_amount=Decimal("999.00"),
                currency="CNY",
            )
        )

    fake_llm = ScriptedOrderLLM(foreign_order_no)
    application.dependency_overrides[get_llm_client] = lambda: fake_llm
    try:
        response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat",
            json={"content": f"查询订单 {foreign_order_no}"},
        )
    finally:
        application.dependency_overrides.pop(get_llm_client, None)

    assert response.status_code == 201
    assert response.json()["content"] == "当前账户下未找到该订单，请核对订单号。"
    assert fake_llm.tool_result == {
        "ok": False,
        "error": {
            "code": "order_not_found",
            "message": "当前客户名下未找到该订单，请核对订单号。",
        },
    }

    async with db_session.begin():
        conversation = await db_session.scalar(
            select(Conversation).where(Conversation.conversation_no == conversation_no)
        )
        assert conversation is not None
        audit = await db_session.scalar(
            select(ToolAuditLog).where(
                ToolAuditLog.conversation_id == conversation.id,
                ToolAuditLog.tool_call_id == "call-order-1",
            )
        )
    assert audit is not None
    assert audit.status == ToolAuditStatus.REJECTED
    assert audit.error_code == "order_not_found"


async def test_order_tool_rejects_invalid_model_arguments(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    async with db_session.begin():
        customer = Customer(
            customer_no=f"CUST-ARGS-{suffix}",
            display_name="参数校验测试用户",
        )
        db_session.add(customer)
        await db_session.flush()
        conversation = Conversation(
            conversation_no=f"CONV-ARGS-{suffix}",
            customer_id=customer.id,
        )
        db_session.add(conversation)
        await db_session.flush()

    result = await OrderToolExecutor(db_session).execute(
        FunctionToolCall(
            id="call-invalid-arguments",
            name="get_order_status",
            arguments='{"order_no":"包含 空格","customer_id":999}',
        ),
        conversation_id=conversation.id,
        customer_id=customer.id,
        idempotency_key=f"conversation:{conversation.id}:sequence:1",
    )

    assert json.loads(result) == {
        "ok": False,
        "error": {
            "code": "invalid_arguments",
            "message": "订单号参数格式无效，请向用户确认完整订单号。",
        },
    }
    async with db_session.begin():
        audit = await db_session.scalar(
            select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "call-invalid-arguments")
        )
    assert audit is not None
    assert audit.status == ToolAuditStatus.REJECTED
    assert audit.error_code == "invalid_arguments"
