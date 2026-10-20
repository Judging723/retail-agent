import pytest
from pydantic import ValidationError

from app.schemas.message import MessageCreateRequest


def test_message_request_strips_whitespace() -> None:
    request = MessageCreateRequest(content="  我的订单什么时候发货？  ")

    assert request.content == "我的订单什么时候发货？"


def test_message_request_rejects_blank_content() -> None:
    with pytest.raises(ValidationError):
        MessageCreateRequest(content="   ")
