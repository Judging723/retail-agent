from model_pipeline.redaction import detect_pii, redact_text


def test_redact_text_removes_high_confidence_identifiers() -> None:
    raw = (
        "手机号13800138000，邮箱buyer@example.com，身份证110101199001011234，"
        "银行卡6222021234567890123，订单号ORDER-123456。"
    )

    result = redact_text(raw)

    assert result.counts == {
        "order_id": 1,
        "email": 1,
        "id_card": 1,
        "mobile": 1,
        "bank_card": 1,
    }
    assert detect_pii(result.text) == {}
    assert "<MOBILE>" in result.text
    assert "<ORDER_ID>" in result.text


def test_redact_text_does_not_change_ordinary_product_numbers() -> None:
    raw = "我要 512G 的手机，预算 5000 元，床宽 1.2m。"

    result = redact_text(raw)

    assert result.text == raw
    assert result.counts == {}
