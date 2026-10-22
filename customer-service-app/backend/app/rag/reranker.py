from collections.abc import Sequence
from typing import Any, Protocol

DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class PairReranker(Protocol):
    """Score each query-document pair; larger scores mean more relevant."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Use a pinned BGE CrossEncoder to compare a query with whole documents."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        revision: str = DEFAULT_RERANKER_MODEL_REVISION,
        batch_size: int = 8,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if model is None:
            import torch
            from sentence_transformers import CrossEncoder

            selected_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            model = CrossEncoder(
                model_name_or_path=model_name,
                revision=revision,
                max_length=512,
                device=selected_device,
            )

        self._model = model
        self._batch_size = batch_size

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []

        scores = self._model.predict(
            [(query, document) for document in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]
