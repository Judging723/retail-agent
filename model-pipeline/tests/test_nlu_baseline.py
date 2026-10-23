import json
from pathlib import Path

import pytest

from model_pipeline.nlu_baseline import evaluate_predictions, train_and_evaluate
from model_pipeline.nlu_dataset import prepare_clarification_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = (
    REPOSITORY_ROOT
    / "knowledge-pipeline"
    / "data"
    / "raw"
    / "chinese_ambiguous_reference"
    / "All Conversation.json"
)
requires_public_raw_data = pytest.mark.skipif(
    not RAW_DATA.exists(),
    reason="run knowledge-pipeline/scripts/prepare_public_knowledge.py first",
)


def test_evaluate_predictions_reports_binary_confusion_matrix() -> None:
    metrics = evaluate_predictions([0, 0, 1, 1], [0, 1, 0, 1])

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


@requires_public_raw_data
def test_tfidf_baseline_beats_majority_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    prepare_clarification_dataset(RAW_DATA, data_dir)

    report = train_and_evaluate(data_dir, tmp_path / "model.joblib")

    for split in ("dev", "test"):
        results = report["results"][split]
        assert results["tfidf_logistic_regression"]["macro_f1"] > results["majority"]["macro_f1"]
    assert json.dumps(report, ensure_ascii=False)
