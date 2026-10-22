from collections.abc import Sequence

import pytest

from app.rag import (
    DEFAULT_RERANKER_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_REVISION,
    CrossEncoderReranker,
)


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs: Sequence[tuple[str, str]] = []
        self.options: dict = {}

    def predict(self, pairs, **kwargs) -> list[float]:
        self.pairs = pairs
        self.options = kwargs
        return [0.2, 0.8]


def test_cross_encoder_scores_query_document_pairs() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model, batch_size=4)

    scores = reranker.score("用户问题", ["候选一", "候选二"])

    assert scores == [0.2, 0.8]
    assert model.pairs == [("用户问题", "候选一"), ("用户问题", "候选二")]
    assert model.options == {
        "batch_size": 4,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


def test_cross_encoder_skips_model_for_empty_documents() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    assert reranker.score("用户问题", []) == []
    assert model.pairs == []


def test_non_positive_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        CrossEncoderReranker(model=FakeCrossEncoder(), batch_size=0)


def test_reranker_model_matches_travel_agent() -> None:
    assert DEFAULT_RERANKER_MODEL_NAME == "BAAI/bge-reranker-v2-m3"
    assert DEFAULT_RERANKER_MODEL_REVISION == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def test_cross_encoder_uses_pinned_model_and_mps(monkeypatch) -> None:
    arguments: dict = {}

    def fake_cross_encoder(**kwargs):
        arguments.update(kwargs)
        return FakeCrossEncoder()

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_cross_encoder)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    CrossEncoderReranker()

    assert arguments == {
        "model_name_or_path": DEFAULT_RERANKER_MODEL_NAME,
        "revision": DEFAULT_RERANKER_MODEL_REVISION,
        "max_length": 512,
        "device": "mps",
    }
