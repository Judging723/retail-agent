from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    Customer,
    Order,
    OrderStatus,
    Refund,
    RefundStatus,
    Shipment,
    ShipmentStatus,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
]


async def test_order_shipment_and_refund_can_be_persisted(
    db_session: AsyncSession,
) -> None:
    suffix = uuid4().hex[:12]
    customer = Customer(
        customer_no=f"CUST-ORDER-{suffix}",
        display_name="订单模型测试用户",
    )
    order = Order(
        order_no=f"ORDER-{suffix}",
        customer=customer,
        status=OrderStatus.SHIPPED,
        total_amount=Decimal("299.00"),
        currency="CNY",
    )
    order.shipment = Shipment(
        carrier="测试快递",
        tracking_no=f"TRACK-{suffix}",
        status=ShipmentStatus.IN_TRANSIT,
        latest_event="包裹已到达测试中转站",
    )
    order.refunds.append(
        Refund(
            refund_no=f"REFUND-{suffix}",
            amount=Decimal("20.00"),
            reason="测试部分退款",
            status=RefundStatus.PROCESSING,
        )
    )

    db_session.add(order)
    await db_session.flush()
    order_id = order.id

    db_session.expunge_all()

    saved_order = await db_session.scalar(
        select(Order)
        .options(
            selectinload(Order.shipment),
            selectinload(Order.refunds),
        )
        .where(Order.id == order_id)
    )

    assert saved_order is not None
    assert saved_order.customer_id == customer.id
    assert saved_order.total_amount == Decimal("299.00")
    assert saved_order.shipment is not None
    assert saved_order.shipment.tracking_no == f"TRACK-{suffix}"
    assert [refund.refund_no for refund in saved_order.refunds] == [f"REFUND-{suffix}"]
