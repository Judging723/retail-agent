import re
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int]


_LABELED_ORDER_ID: Final = re.compile(
    r"(?P<label>(?:订单号|运单号|快递单号)\s*[：:]?\s*)"
    r"(?P<value>[A-Za-z0-9_-]{6,})",
    re.IGNORECASE,
)

_PII_PATTERNS: Final[tuple[tuple[str, re.Pattern[str], str], ...]] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"),
        "<EMAIL>",
    ),
    (
        "id_card",
        re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)"),
        "<ID_CARD>",
    ),
    (
        "mobile",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "<MOBILE>",
    ),
    (
        "bank_card",
        re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
        "<BANK_CARD>",
    ),
)


def redact_text(text: str) -> RedactionResult:
    """Replace high-confidence identifiers and report how many were removed."""

    counts: dict[str, int] = {}

    def replace_order_id(match: re.Match[str]) -> str:
        counts["order_id"] = counts.get("order_id", 0) + 1
        return f"{match.group('label')}<ORDER_ID>"

    redacted = _LABELED_ORDER_ID.sub(replace_order_id, text)
    for name, pattern, replacement in _PII_PATTERNS:
        redacted, count = pattern.subn(replacement, redacted)
        if count:
            counts[name] = counts.get(name, 0) + count

    return RedactionResult(text=redacted, counts=counts)


def detect_pii(text: str) -> dict[str, int]:
    """Return high-confidence PII matches that remain in text."""

    counts: dict[str, int] = {}
    order_count = len(_LABELED_ORDER_ID.findall(text))
    if order_count:
        counts["order_id"] = order_count

    for name, pattern, _ in _PII_PATTERNS:
        count = len(pattern.findall(text))
        if count:
            counts[name] = count
    return counts
