import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ENTITY_NAMES = ("order_no", "amount", "reason")

_ORDER_CONTEXT_PATTERN = re.compile(
    r"(?:订单号|订单)\s*(?:是|为|[:：])?\s*([A-Za-z0-9][A-Za-z0-9_-]{2,31})",
    re.IGNORECASE,
)
_STRUCTURED_ORDER_PATTERN = re.compile(
    r"\b(?=[A-Za-z0-9_-]{3,32}\b)(?=[A-Za-z0-9_-]*[A-Za-z])"
    r"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b",
    re.IGNORECASE,
)
_AMOUNT_SUFFIX_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?P<amount>\d+(?:\.\d{1,2})?)\s*元")
_AMOUNT_PREFIX_PATTERN = re.compile(r"[¥￥]\s*(?P<amount>\d+(?:\.\d{1,2})?)")
_REASON_MARKER_PATTERN = re.compile(r"原因\s*(?:是|写(?:成|为)?|[:：])\s*([^，。；！？,;!?]+)")
_REASON_AFTER_AMOUNT_PATTERN = re.compile(
    r"(?:退款|退)\s*(?:[¥￥]\s*)?\d+(?:\.\d{1,2})?\s*元\s*[，,]\s*"
    r"(?!原因)([^。！？!?]+)"
)


def extract_entities(text: str) -> dict[str, str]:
    """Extract only the three fields required by the project's current order tools."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    entities: dict[str, str] = {}

    order_match = _ORDER_CONTEXT_PATTERN.search(text) or _STRUCTURED_ORDER_PATTERN.search(text)
    if order_match:
        entities["order_no"] = (
            order_match.group(1) if order_match.lastindex else order_match.group(0)
        )

    amount = _extract_amount(text)
    if amount is not None:
        entities["amount"] = amount

    reason_match = _REASON_MARKER_PATTERN.search(text) or _REASON_AFTER_AMOUNT_PATTERN.search(text)
    if reason_match:
        reason = reason_match.group(1).strip()
        if len(reason) >= 2:
            entities["reason"] = reason

    return entities


def evaluate_entity_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate exact field values with per-entity and micro precision/recall/F1."""

    if not cases:
        raise ValueError("evaluation cases must not be empty")
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ENTITY_NAMES}
    predictions: list[dict[str, Any]] = []
    exact_matches = 0

    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        text = case.get("text")
        expected = case.get("expected")
        if not isinstance(case_id, str) or not isinstance(text, str):
            raise TypeError(f"case {index}: case_id and text must be strings")
        if not isinstance(expected, dict) or set(expected) - set(ENTITY_NAMES):
            raise ValueError(f"{case_id}: expected contains an unknown entity")

        predicted = extract_entities(text)
        exact_matches += predicted == expected
        predictions.append(
            {
                "case_id": case_id,
                "expected": expected,
                "predicted": predicted,
                "exact_match": predicted == expected,
            }
        )
        for entity_name in ENTITY_NAMES:
            expected_value = expected.get(entity_name)
            predicted_value = predicted.get(entity_name)
            if expected_value == predicted_value and expected_value is not None:
                counts[entity_name]["tp"] += 1
            elif expected_value != predicted_value:
                if predicted_value is not None:
                    counts[entity_name]["fp"] += 1
                if expected_value is not None:
                    counts[entity_name]["fn"] += 1

    per_entity = {name: _metrics(**counts[name]) for name in ENTITY_NAMES}
    totals = {
        key: sum(entity_counts[key] for entity_counts in counts.values())
        for key in ("tp", "fp", "fn")
    }
    return {
        "format_version": 1,
        "task": "rule-based extraction for current order-tool arguments",
        "entities": list(ENTITY_NAMES),
        "cases": len(cases),
        "exact_match_accuracy": round(exact_matches / len(cases), 6),
        "micro": _metrics(**totals),
        "per_entity": per_entity,
        "predictions": predictions,
        "limitations": (
            "The cases are a small hand-written regression set. The extractor supports one order number, "
            "one amount, and an explicitly phrased refund reason; online Pydantic validation remains the "
            "authority for tool arguments."
        ),
    }


def load_entity_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("entity evaluation file must contain a JSON list")
    return payload


def _extract_amount(text: str) -> str | None:
    matches = [
        (match.start(), match.group("amount"))
        for pattern in (_AMOUNT_PREFIX_PATTERN, _AMOUNT_SUFFIX_PATTERN)
        for match in pattern.finditer(text)
    ]
    if not matches:
        return None
    raw_amount = min(matches, key=lambda item: item[0])[1]
    try:
        return f"{Decimal(raw_amount):.2f}"
    except InvalidOperation as exc:
        raise ValueError(f"invalid amount: {raw_amount}") from exc


def _metrics(*, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
