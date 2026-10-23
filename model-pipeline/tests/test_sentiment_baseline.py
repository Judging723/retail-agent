from pathlib import Path

import pytest

from model_pipeline.sentiment_baseline import (
    evaluate_sentiment_predictions,
    lexicon_predict,
    train_and_evaluate_sentiment,
)
from model_pipeline.sentiment_dataset import prepare_sentiment_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kapibala_sales_dialogues"
META_PATH = RAW_DIR / "meta.jsonl"
CONVERSATIONS_PATH = RAW_DIR / "conversations_L3.jsonl"
requires_public_raw_data = pytest.mark.skipif(
    not META_PATH.exists() or not CONVERSATIONS_PATH.exists(),
    reason="download the pinned Kapibala source files first",
)


def test_sentiment_metrics_keep_explicit_label_order() -> None:
    metrics = evaluate_sentiment_predictions(
        ["negative", "neutral", "positive"],
        ["negative", "positive", "positive"],
    )

    assert metrics["accuracy"] == 0.666667
    assert metrics["label_order"] == ["negative", "neutral", "positive"]
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_lexicon_rule_has_a_neutral_fallback() -> None:
    assert lexicon_predict("这个商品太差了，我要退款") == "negative"
    assert lexicon_predict("很好，谢谢你") == "positive"
    assert lexicon_predict("请介绍一下产品参数") == "neutral"


@requires_public_raw_data
def test_tfidf_sentiment_baseline_beats_majority_macro_f1(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    prepare_sentiment_dataset(META_PATH, CONVERSATIONS_PATH, data_dir)

    report = train_and_evaluate_sentiment(data_dir, tmp_path / "model.joblib")

    for split in ("dev", "test"):
        results = report["results"][split]
        assert results["tfidf_logistic_regression"]["macro_f1"] > results["majority"]["macro_f1"]
