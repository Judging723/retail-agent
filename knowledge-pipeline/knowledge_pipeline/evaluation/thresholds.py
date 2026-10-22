from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    balanced_accuracy: float
    positive_recall: float
    negative_rejection: float


def select_relevance_threshold(
    labeled_scores: Sequence[tuple[float, bool]],
) -> ThresholdSelection:
    """Select a threshold using only a development set of labeled top scores."""

    positive_scores = [score for score, is_positive in labeled_scores if is_positive]
    negative_scores = [score for score, is_positive in labeled_scores if not is_positive]
    if not positive_scores or not negative_scores:
        raise ValueError("threshold calibration requires positive and negative scores")

    unique_scores = sorted({score for score, _ in labeled_scores})
    candidates = [0.0]
    candidates.extend((left + right) / 2 for left, right in pairwise(unique_scores))
    candidates.append(1.0)

    selections = [
        _evaluate_threshold(value, positive_scores, negative_scores) for value in candidates
    ]
    return max(
        selections,
        key=lambda item: (
            item.balanced_accuracy,
            item.negative_rejection,
            item.positive_recall,
            item.threshold,
        ),
    )


def _evaluate_threshold(
    threshold: float,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> ThresholdSelection:
    positive_recall = sum(score >= threshold for score in positive_scores) / len(positive_scores)
    negative_rejection = sum(score < threshold for score in negative_scores) / len(negative_scores)
    return ThresholdSelection(
        threshold=threshold,
        balanced_accuracy=(positive_recall + negative_rejection) / 2,
        positive_recall=positive_recall,
        negative_rejection=negative_rejection,
    )
