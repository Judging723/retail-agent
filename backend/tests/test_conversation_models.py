from sqlalchemy import UniqueConstraint

from app.models import (
    Conversation,
    ConversationStatus,
    Customer,
    Message,
    MessageRole,
)


def test_conversation_columns_and_foreign_key() -> None:
    columns = Conversation.__table__.c

    assert set(columns.keys()) == {
        "id",
        "conversation_no",
        "customer_id",
        "status",
        "summary",
        "created_at",
        "updated_at",
    }

    customer_fk = next(iter(columns.customer_id.foreign_keys))

    assert customer_fk.target_fullname == "customers.id"
    assert customer_fk.ondelete == "RESTRICT"
    assert columns.customer_id.nullable is False
    assert columns.customer_id.index is True


def test_message_columns_and_foreign_key() -> None:
    columns = Message.__table__.c

    assert set(columns.keys()) == {
        "id",
        "conversation_id",
        "sequence_no",
        "role",
        "content",
        "created_at",
    }

    conversation_fk = next(iter(columns.conversation_id.foreign_keys))

    assert conversation_fk.target_fullname == "conversations.id"
    assert conversation_fk.ondelete == "CASCADE"


def test_message_sequence_is_unique_per_conversation() -> None:
    unique_column_groups = {
        tuple(constraint.columns.keys())
        for constraint in Message.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert ("conversation_id", "sequence_no") in unique_column_groups


def test_model_relationships_are_bidirectional() -> None:
    assert Customer.conversations.property.back_populates == "customer"
    assert Conversation.customer.property.back_populates == "conversations"
    assert Conversation.messages.property.back_populates == "conversation"
    assert Message.conversation.property.back_populates == "messages"


def test_domain_enum_values() -> None:
    assert ConversationStatus.OPEN.value == "open"
    assert ConversationStatus.WAITING_FOR_HUMAN.value == "waiting_for_human"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"