import re

POLICY_FALLBACK_RESPONSE = (
    "这个问题需要以平台的权威规则以及具体订单或商品状态为准。"
    "当前知识库没有对应的官方政策，我不能提供具体期限、金额或承诺；"
    "请查看订单页规则或联系人工客服核实。"
)

_POLICY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"(退款|退货).*(多久|几天|期限|时效|到账)",
        r"(多久|几天|期限|时效).*(退款|退货|到账)",
        r"七天无理由|无理由退货",
        r"退货运费|运费.*(谁承担|谁出|怎么收)",
        r"(赔付|赔偿|补偿).*(多少|金额|规则|条件|一定|会不会)",
        r"(超时|延误|破损|丢失).*(赔付|赔偿|补偿)",
        r"(保修|质保).*(多久|几年|期限)",
        r"换货.*(多久|几天|期限|时效)",
        r"发票.*(多久|几天|开出|时效)",
        r"优惠券.*(过期|恢复)",
        r"价格保护|保价|补差价",
        r"(预售|定金).*(能退|可以退|退款)",
    )
)


def policy_fallback(user_content: str) -> str | None:
    """Return a deterministic response for policies absent from the current knowledge base."""

    normalized = re.sub(r"\s+", "", user_content)
    if any(pattern.search(normalized) for pattern in _POLICY_PATTERNS):
        return POLICY_FALLBACK_RESPONSE
    return None
