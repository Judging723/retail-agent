from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.customer import Customer
    from app.models.message import Message


class ConversationStatus(StrEnum):
    OPEN = "open"
    WAITING_FOR_HUMAN = "waiting_for_human"
    CLOSED = "closed"


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    conversation_no: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        unique=True,
    )
    customer_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("customers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(
            ConversationStatus,
            name="conversation_status",
            native_enum=False,
            create_constraint=True,
            length=32,
            values_callable=lambda enum_type: [
                item.value for item in enum_type
            ],
        ),
        nullable=False,
        default=ConversationStatus.OPEN,
        server_default=ConversationStatus.OPEN.value,
    )
    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    customer: Mapped[Customer] = relationship(
        back_populates="conversations",
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.sequence_no",
    )