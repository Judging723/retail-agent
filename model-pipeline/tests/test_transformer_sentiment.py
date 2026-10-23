import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from model_pipeline.transformer_sentiment import balanced_class_weights


def test_balanced_class_weights_upweight_rare_labels() -> None:
    weights = balanced_class_weights([0, 0, 0, 1, 1, 2], class_count=3)

    assert weights.tolist() == pytest.approx([2 / 3, 1.0, 2.0])


def test_balanced_class_weights_require_every_class() -> None:
    with pytest.raises(ValueError, match="every configured class"):
        balanced_class_weights([0, 0, 1], class_count=3)
