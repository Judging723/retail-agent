from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class TextEmbedder(Protocol):
    """Contract used by semantic retrieval, independent of a model library."""

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Sentence Transformers adapter for the pinned Chinese BGE model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        revision: str = DEFAULT_MODEL_REVISION,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
        )

    @property
    def device(self) -> str:
        return str(self._model.device)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        instructed_queries = [f"{QUERY_INSTRUCTION}{text}" for text in texts]
        return self._encode(instructed_queries)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
