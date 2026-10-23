import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from model_pipeline.sentiment_inference import emotion_risk


@pytest.mark.parametrize(
    ("text", "sentiment", "negative_probability", "expected_level"),
    [
        ("我非常生气，要投诉你们", "negative", 0.55, "high"),
        ("商品好像有问题", "negative", 0.55, "medium"),
        ("我有一点担心", "positive", 0.42, "medium"),
        ("请介绍商品参数", "neutral", 0.2, "low"),
    ],
)
def test_emotion_risk_levels(
    text: str,
    sentiment: str,
    negative_probability: float,
    expected_level: str,
) -> None:
    level, hint = emotion_risk(
        text=text,
        sentiment=sentiment,
        negative_probability=negative_probability,
    )

    assert level == expected_level
    assert hint.endswith("。")


def test_emotion_risk_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        emotion_risk(text="测试", sentiment="negative", negative_probability=1.1)
