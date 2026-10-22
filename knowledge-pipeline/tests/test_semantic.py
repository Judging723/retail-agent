from collections.abc import Sequence

import pytest

from knowledge_pipeline.ingestion.schemas import KnowledgeDocument
from knowledge_pipeline.retrieval.semantic import SemanticRetriever


class FakeEmbedder:
    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        assert list(texts) == ["甲文档：退货规则", "乙文档：物流规则"]
        return [[1.0, 0.0], [0.0, 1.0]]

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        assert list(texts) == ["我要退货"]
        return [[0.9, 0.1]]


def test_semantic_retriever_ranks_by_vector_similarity() -> None:
    retriever = SemanticRetriever(
        [
            _make_document("doc-a", title="甲文档", content="退货规则"),
            _make_document("doc-b", title="乙文档", content="物流规则"),
        ],
        FakeEmbedder(),
    )

    results = retriever.search("我要退货", limit=2)

    assert [result.source_id for result in results] == ["doc-a", "doc-b"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.1)


@pytest.mark.parametrize(("query", "limit"), [("", 1), ("   ", 1), ("退货", 0)])
def test_invalid_search_input_is_rejected(query: str, limit: int) -> None:
    retriever = SemanticRetriever([], FakeEmbedder())

    with pytest.raises(ValueError):
        retriever.search(query, limit=limit)


def _make_document(source_id: str, *, title: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        source_id=source_id,
        title=title,
        category="test",
        content=content,
        keywords=(),
        revision="test-revision",
        source_date="2024-04-21",
        source_url="https://example.com/dataset",
        license="MIT",
    )
