from collections.abc import Sequence
from dataclasses import dataclass

from knowledge_pipeline.embeddings import TextEmbedder
from knowledge_pipeline.schemas import KnowledgeDocument


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    source_id: str
    score: float


class SemanticRetriever:
    """In-memory dense retriever used to establish a semantic-search baseline."""

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        embedder: TextEmbedder,
    ) -> None:
        self._documents = tuple(documents)
        self._embedder = embedder
        document_texts = [f"{document.title}：{document.content}" for document in self._documents]
        self._document_embeddings = tuple(
            tuple(embedding) for embedding in embedder.encode_documents(document_texts)
        )

        if len(self._document_embeddings) != len(self._documents):
            raise ValueError("embedder must return one embedding per document")

    def search(self, query: str, *, limit: int = 3) -> list[SemanticSearchResult]:
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not self._documents:
            return []

        query_embeddings = self._embedder.encode_queries([query.strip()])
        if len(query_embeddings) != 1:
            raise ValueError("embedder must return exactly one query embedding")

        query_embedding = query_embeddings[0]
        scored_results = [
            SemanticSearchResult(
                source_id=document.source_id,
                score=_dot_product(query_embedding, document_embedding),
            )
            for document, document_embedding in zip(
                self._documents,
                self._document_embeddings,
                strict=True,
            )
        ]
        scored_results.sort(key=lambda result: (-result.score, result.source_id))
        return scored_results[:limit]


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and document embeddings must have the same dimensions")
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
