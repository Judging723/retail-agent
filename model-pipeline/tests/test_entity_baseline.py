from pathlib import Path

from model_pipeline.entity_baseline import (
    evaluate_entity_cases,
    extract_entities,
    load_entity_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "data" / "entity_cases.json"


def test_extract_entities_normalizes_current_tool_arguments() -> None:
    assert extract_entities("请给订单 ORDER-20001 申请退款￥18.5，原因是商品破损。") == {
        "order_no": "ORDER-20001",
        "amount": "18.50",
        "reason": "商品破损",
    }


def test_entity_baseline_passes_fixed_regression_cases() -> None:
    report = evaluate_entity_cases(load_entity_cases(CASES_PATH))

    assert report["cases"] == 13
    assert report["exact_match_accuracy"] == 1.0
    assert report["micro"]["f1"] == 1.0
