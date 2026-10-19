import pytest
from pydantic import ValidationError

from app.schemas.conversation import ConversationCreateRequest


def test_create_request_strips_whitespace() -> None:
    request = ConversationCreateRequest(
        customer_no="  CUST-001  ",
        display_name="  小王  ",
    )
    assert request.customer_no == "CUST-001"
    assert request.display_name == "小王"


def test_create_request_allows_omitted_display_name() -> None:
    request = ConversationCreateRequest(
        customer_no="CUST-001",
    )
    assert request.display_name is None


def test_create_request_allows_explicit_none_display_name() -> None:
    request = ConversationCreateRequest(
        customer_no="CUST-001",
        display_name=None,
    )
    assert request.display_name is None


def test_create_request_rejects_invalid_customer_no() -> None:
    with pytest.raises(ValidationError):
        ConversationCreateRequest(
            customer_no="客户 001",
        )
