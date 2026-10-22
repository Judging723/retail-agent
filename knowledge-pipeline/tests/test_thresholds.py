import pytest

from knowledge_pipeline.evaluation.thresholds import select_relevance_threshold


def test_threshold_separates_positive_and_negative_scores() -> None:
    selection = select_relevance_threshold(
        [
            (0.72, True),
            (0.64, True),
            (0.41, False),
            (0.33, False),
        ]
    )

    assert selection.threshold == pytest.approx(0.525)
    assert selection.balanced_accuracy == 1.0
    assert selection.positive_recall == 1.0
    assert selection.negative_rejection == 1.0


@pytest.mark.parametrize(
    "scores",
    [
        [(0.7, True)],
        [(0.3, False)],
        [],
    ],
)
def test_threshold_requires_both_classes(scores: list[tuple[float, bool]]) -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        select_relevance_threshold(scores)
