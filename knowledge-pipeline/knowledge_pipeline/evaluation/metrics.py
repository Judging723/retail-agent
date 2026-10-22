import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    query: str
    relevant_source_ids: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    positive_cases: int
    negative_cases: int
    hit_at_k: float
    mrr: float
    negative_accuracy: float


SearchFunction = Callable[[str, int], Sequence[str]]


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Load and validate a versioned retrieval evaluation set."""

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("evaluation file must contain a JSON object")

    raw_cases = raw_data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("evaluation file must contain a 'cases' list")

    cases: list[RetrievalCase] = []
    seen_case_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        case = _parse_case(raw_case, index=index)
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")

        seen_case_ids.add(case.case_id)
        cases.append(case)

    return cases


def evaluate_retrieval(
    cases: Sequence[RetrievalCase],
    search: SearchFunction,
    *,
    k: int = 3,
) -> RetrievalMetrics:
    """Evaluate any retriever that returns ranked source IDs."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    positive_cases = 0
    negative_cases = 0
    hits = 0
    reciprocal_rank_total = 0.0
    correct_negatives = 0

    for case in cases:
        retrieved_ids = tuple(search(case.query, k))[:k]
        relevant_ids = set(case.relevant_source_ids)

        if relevant_ids:
            positive_cases += 1
            first_relevant_rank = next(
                (
                    rank
                    for rank, source_id in enumerate(retrieved_ids, start=1)
                    if source_id in relevant_ids
                ),
                None,
            )
            if first_relevant_rank is not None:
                hits += 1
                reciprocal_rank_total += 1 / first_relevant_rank
        else:
            negative_cases += 1
            if not retrieved_ids:
                correct_negatives += 1

    return RetrievalMetrics(
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        hit_at_k=hits / positive_cases if positive_cases else 0.0,
        mrr=reciprocal_rank_total / positive_cases if positive_cases else 0.0,
        negative_accuracy=correct_negatives / negative_cases if negative_cases else 0.0,
    )


def _parse_case(raw_case: Any, *, index: int) -> RetrievalCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"cases[{index}] must be a JSON object")

    case_id = _required_string(raw_case, field="case_id", index=index)
    query = _required_string(raw_case, field="query", index=index)
    relevant_source_ids = _string_tuple(
        raw_case.get("relevant_source_ids"),
        field="relevant_source_ids",
        index=index,
    )
    tags = _string_tuple(raw_case.get("tags"), field="tags", index=index)

    return RetrievalCase(
        case_id=case_id,
        query=query,
        relevant_source_ids=relevant_source_ids,
        tags=tags,
    )


def _required_string(raw_case: dict[str, Any], *, field: str, index: int) -> str:
    value = raw_case.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"cases[{index}].{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, *, field: str, index: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"cases[{index}].{field} must be a list")

    items: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"cases[{index}].{field}[{item_index}] must be a non-empty string")
        items.append(item.strip())
    return tuple(items)
