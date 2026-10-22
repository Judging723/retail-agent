from app.models import (
    Base,
    Customer,
    Order,
    OrderStatus,
    Refund,
    RefundStatus,
    Shipment,
    ShipmentStatus,
)


def test_business_tables_are_registered() -> None:
    assert {"orders", "shipments", "refunds"} <= set(Base.metadata.tables)


def test_order_columns_and_customer_foreign_key() -> None:
    assert set(Order.__table__.c.keys()) == {
        "id",
        "order_no",
        "customer_id",
        "status",
        "total_amount",
        "currency",
        "created_at",
        "updated_at",
    }

    foreign_key = next(iter(Order.__table__.c.customer_id.foreign_keys))
    assert foreign_key.target_fullname == "customers.id"


def test_shipment_columns_and_order_foreign_key() -> None:
    assert set(Shipment.__table__.c.keys()) == {
        "id",
        "order_id",
        "carrier",
        "tracking_no",
        "status",
        "latest_event",
        "created_at",
        "updated_at",
    }

    foreign_key = next(iter(Shipment.__table__.c.order_id.foreign_keys))
    assert foreign_key.target_fullname == "orders.id"
    assert Shipment.__table__.c.order_id.unique is True


def test_refund_columns_and_order_foreign_key() -> None:
    assert set(Refund.__table__.c.keys()) == {
        "id",
        "refund_no",
        "idempotency_key",
        "order_id",
        "amount",
        "reason",
        "status",
        "created_at",
        "updated_at",
    }

    foreign_key = next(iter(Refund.__table__.c.order_id.foreign_keys))
    assert foreign_key.target_fullname == "orders.id"


def test_business_model_relationships_are_bidirectional() -> None:
    assert Customer.orders.property.back_populates == "customer"
    assert Order.customer.property.back_populates == "orders"
    assert Order.shipment.property.back_populates == "order"
    assert Shipment.order.property.back_populates == "shipment"
    assert Order.refunds.property.back_populates == "order"
    assert Refund.order.property.back_populates == "refunds"


def test_business_status_values() -> None:
    assert [status.value for status in OrderStatus] == [
        "pending_payment",
        "paid",
        "fulfilling",
        "shipped",
        "completed",
        "canceled",
        "refunded",
    ]
    assert [status.value for status in ShipmentStatus] == [
        "pending",
        "in_transit",
        "delivered",
        "exception",
    ]
    assert [status.value for status in RefundStatus] == [
        "requested",
        "processing",
        "succeeded",
        "rejected",
        "failed",
    ]
