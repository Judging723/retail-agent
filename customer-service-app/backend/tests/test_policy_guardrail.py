from app.services.policy_guardrail import POLICY_FALLBACK_RESPONSE, policy_fallback


def test_policy_questions_use_deterministic_fallback() -> None:
    questions = [
        "退款一般几天到账？",
        "拆封以后还能七天无理由退货吗？",
        "快递超时一定会赔偿吗？",
        "这个商品保修几年？",
        "预售定金可以退款吗？",
    ]

    assert [policy_fallback(question) for question in questions] == [
        POLICY_FALLBACK_RESPONSE
    ] * len(questions)


def test_operational_requests_are_not_blocked_as_policy_questions() -> None:
    requests = [
        "帮我查询订单 ORDER-10001。",
        "请为订单 ORDER-10001 退款 50 元，原因是商品破损。",
        "我想买一件黑色 M 码防晒衣。",
    ]

    assert all(policy_fallback(request) is None for request in requests)
