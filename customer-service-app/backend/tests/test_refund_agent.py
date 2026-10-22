import json
from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import OrderToolExecutor
from app.api.dependencies import get_llm_client
from app.llm import AssistantTurn, ChatMessage, FunctionToolCall, ToolDefinition
from app.models import (
    Conversation,
    Customer,
    Order,
    OrderStatus,
    Refund,
    RefundStatus,
    ToolAuditLog,
    ToolAuditStatus,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


class ScriptedRefundLLM:
    def __init__(self, *, order_no: str, amount: str, reason: str) -> None:
        self.order_no = order_no
        self.amount = amount
        self.reason = reason
        self.tool_result: dict | None = None

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("Agent should use complete_with_tools")

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        if messages[-1]["role"] != "tool":
            assert {tool["function"]["name"] for tool in tools} == {
                "get_order_status",
                "request_refund",
            }
            return AssistantTurn(
                content=None,
                tool_calls=(
                    FunctionToolCall(
                        id="call-refund-1",
                        name="request_refund",
                        arguments=json.dumps(
                            {
                                "order_no": self.order_no,
                                "amount": self.amount,
                                "reason": self.reason,
                            }
                        ),
                    ),
                ),
            )

        self.tool_result = json.loads(messages[-1]["content"])
        if self.tool_result["ok"]:
            refund = self.tool_result["refund"]
            return AssistantTurn(
                content=(
                    f"退款申请 {refund['refund_no']} 已提交，"
                    f"金额 {refund['amount']} {refund['currency']}。"
                ),
                tool_calls=(),
            )
        return AssistantTurn(
            content=self.tool_result["error"]["message"],
            tool_calls=(),
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        if not messages:
            yield ""
        raise AssertionError("This test does not use direct model streaming")


async def _create_customer_conversation_order(
    db_session: AsyncSession,
    *,
    suffix: str,
    total_amount: str = "200.00",
) -> tuple[Customer, Conversation, Order]:
    async with db_session.begin():
        customer = Customer(
            customer_no=f"CUST-REFUND-{suffix}",
            display_name="退款测试用户",
        )
        db_session.add(customer)
        await db_session.flush()
        conversation = Conversation(
            conversation_no=f"CONV-REFUND-{suffix}",
            customer_id=customer.id,
        )
        order = Order(
            order_no=f"ORDER-REFUND-{suffix}",
            customer_id=customer.id,
            status=OrderStatus.SHIPPED,
            total_amount=Decimal(total_amount),
            currency="CNY",
        )
        db_session.add_all([conversation, order])
        await db_session.flush()
    return customer, conversation, order


def _refund_call(
    *,
    call_id: str,
    order_no: str,
    amount: str,
    reason: str = "商品破损",
) -> FunctionToolCall:
    return FunctionToolCall(
        id=call_id,
        name="request_refund",
        arguments=json.dumps(
            {
                "order_no": order_no,
                "amount": amount,
                "reason": reason,
            }
        ),
    )


async def test_refund_agent_creates_requested_refund_and_audit(
    application: FastAPI,
    api_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer_no = f"CUST-API-REFUND-{suffix}"
    order_no = f"ORDER-{suffix}"
    create_response = await api_client.post(
        "/api/v1/conversations",
        json={"customer_no": customer_no, "display_name": "退款 API 用户"},
    )
    conversation_no = create_response.json()["conversation_no"]
    async with db_session.begin():
        customer = await db_session.scalar(
            select(Customer).where(Customer.customer_no == customer_no)
        )
        assert customer is not None
        db_session.add(
            Order(
                order_no=order_no,
                customer_id=customer.id,
                status=OrderStatus.SHIPPED,
                total_amount=Decimal("299.00"),
                currency="CNY",
            )
        )

    fake_llm = ScriptedRefundLLM(
        order_no=order_no,
        amount="80.00",
        reason="收到的商品外壳破损",
    )
    application.dependency_overrides[get_llm_client] = lambda: fake_llm
    try:
        response = await api_client.post(
            f"/api/v1/conversations/{conversation_no}/chat",
            headers={"Idempotency-Key": f"refund-request-{suffix}"},
            json={"content": f"请为订单 {order_no} 申请退款 80 元，商品外壳破损。"},
        )
    finally:
        application.dependency_overrides.pop(get_llm_client, None)

    assert response.status_code == 201
    assert fake_llm.tool_result is not None
    assert fake_llm.tool_result["refund"]["status"] == "requested"
    assert fake_llm.tool_result["refund"]["idempotent_replay"] is False

    async with db_session.begin():
        refund = await db_session.scalar(
            select(Refund).join(Order).where(Order.order_no == order_no)
        )
        audit = await db_session.scalar(
            select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "call-refund-1")
        )
    assert refund is not None
    assert refund.amount == Decimal("80.00")
    assert refund.status == RefundStatus.REQUESTED
    assert refund.idempotency_key == f"refund-request-{suffix}"
    assert audit is not None
    assert audit.status == ToolAuditStatus.SUCCESS


async def test_refund_tool_reuses_same_idempotency_key(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer, conversation, order = await _create_customer_conversation_order(
        db_session,
        suffix=suffix,
    )
    executor = OrderToolExecutor(db_session)
    idempotency_key = f"refund-idempotent-{suffix}"

    first = json.loads(
        await executor.execute(
            _refund_call(
                call_id="call-refund-first",
                order_no=order.order_no,
                amount="50.00",
            ),
            conversation_id=conversation.id,
            customer_id=customer.id,
            idempotency_key=idempotency_key,
        )
    )
    replay = json.loads(
        await executor.execute(
            _refund_call(
                call_id="call-refund-replay",
                order_no=order.order_no,
                amount="50.00",
            ),
            conversation_id=conversation.id,
            customer_id=customer.id,
            idempotency_key=idempotency_key,
        )
    )

    assert replay["refund"]["refund_no"] == first["refund"]["refund_no"]
    assert first["refund"]["idempotent_replay"] is False
    assert replay["refund"]["idempotent_replay"] is True
    async with db_session.begin():
        refund_count = await db_session.scalar(
            select(func.count()).select_from(Refund).where(Refund.order_id == order.id)
        )
    assert refund_count == 1


async def test_refund_tool_rejects_foreign_order_and_excess_amount(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer, conversation, own_order = await _create_customer_conversation_order(
        db_session,
        suffix=suffix,
        total_amount="100.00",
    )
    async with db_session.begin():
        foreign_customer = Customer(
            customer_no=f"CUST-OTHER-{suffix}",
            display_name="其他客户",
        )
        db_session.add(foreign_customer)
        await db_session.flush()
        foreign_order = Order(
            order_no=f"ORDER-OTHER-{suffix}",
            customer_id=foreign_customer.id,
            status=OrderStatus.SHIPPED,
            total_amount=Decimal("500.00"),
            currency="CNY",
        )
        db_session.add(foreign_order)
        db_session.add(
            Refund(
                refund_no=f"REFUND-EXISTING-{suffix}",
                order_id=own_order.id,
                amount=Decimal("40.00"),
                reason="已有退款",
                status=RefundStatus.PROCESSING,
            )
        )

    executor = OrderToolExecutor(db_session)
    foreign_result = json.loads(
        await executor.execute(
            _refund_call(
                call_id="call-refund-foreign",
                order_no=foreign_order.order_no,
                amount="10.00",
            ),
            conversation_id=conversation.id,
            customer_id=customer.id,
            idempotency_key=f"refund-foreign-{suffix}",
        )
    )
    excess_result = json.loads(
        await executor.execute(
            _refund_call(
                call_id="call-refund-excess",
                order_no=own_order.order_no,
                amount="70.00",
            ),
            conversation_id=conversation.id,
            customer_id=customer.id,
            idempotency_key=f"refund-excess-{suffix}",
        )
    )

    assert foreign_result["error"]["code"] == "order_not_found"
    assert excess_result == {
        "ok": False,
        "error": {
            "code": "amount_exceeds_refundable",
            "message": "本订单当前最多可申请退款 60.00 CNY。",
        },
    }
