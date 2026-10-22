import json
from pathlib import Path

import pytest

from knowledge_pipeline.baseline import keyword_search
from knowledge_pipeline.evaluation import (
    RetrievalCase,
    evaluate_retrieval,
    load_retrieval_cases,
)
from knowledge_pipeline.loader import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"
KNOWLEDGE_PATH = REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"


def test_retrieval_metrics() -> None:
    cases = [
        _make_case("first", relevant_source_ids=("doc-a",)),
        _make_case("second", relevant_source_ids=("doc-b",)),
        _make_case("negative", relevant_source_ids=()),
    ]
    rankings = {
        "first": ["doc-a", "doc-c"],
        "second": ["doc-c", "doc-b"],
        "negative": [],
    }

    metrics = evaluate_retrieval(cases, lambda query, k: rankings[query][:k], k=2)

    assert metrics.positive_cases == 2
    assert metrics.negative_cases == 1
    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == 0.75
    assert metrics.negative_accuracy == 1.0


def test_missed_positive_case_has_zero_score() -> None:
    cases = [_make_case("missed", relevant_source_ids=("relevant",))]

    metrics = evaluate_retrieval(cases, lambda _query, _k: ["irrelevant"], k=1)

    assert metrics.hit_at_k == 0.0
    assert metrics.mrr == 0.0


def test_non_positive_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        evaluate_retrieval([], lambda _query, _k: [], k=0)


def test_fixed_evaluation_set_can_be_loaded() -> None:
    cases = load_retrieval_cases(EVALUATION_PATH)

    assert len(cases) == 12
    assert sum(bool(case.relevant_source_ids) for case in cases) == 10
    assert sum(not case.relevant_source_ids for case in cases) == 2


def test_duplicate_case_id_is_rejected(tmp_path: Path) -> None:
    raw_data = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    raw_data["cases"][1]["case_id"] = raw_data["cases"][0]["case_id"]
    path = tmp_path / "duplicate_cases.json"
    path.write_text(json.dumps(raw_data, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_retrieval_cases(path)


def test_keyword_baseline_on_fixed_evaluation_set() -> None:
    documents = load_documents(KNOWLEDGE_PATH)
    cases = load_retrieval_cases(EVALUATION_PATH)

    metrics = evaluate_retrieval(
        cases,
        lambda query, k: keyword_search(documents, query, limit=k),
        k=3,
    )

    assert metrics.positive_cases == 10
    assert metrics.negative_cases == 2
    assert metrics.hit_at_k == 0.4
    assert metrics.mrr == 0.4
    assert metrics.negative_accuracy == 1.0


def _make_case(case_id: str, *, relevant_source_ids: tuple[str, ...]) -> RetrievalCase:
    return RetrievalCase(
        case_id=case_id,
        query=case_id,
        relevant_source_ids=relevant_source_ids,
        tags=("test",),
    )
