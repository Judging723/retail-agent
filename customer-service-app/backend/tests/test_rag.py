from collections.abc import Sequence

import pytest

from app.rag import BM25Document, BM25Retriever, KnowledgeRetriever


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[0.9, 0.1] for _ in texts]


class FakeMilvusClient:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, **kwargs) -> list[list[dict]]:
        self.search_calls.append(kwargs)
        return [
            [
                _hit("car-apple", "苹果", "苹果选购对话", 0.91),
                _hit("car-apple", "苹果", "同一案例的另一片段", 0.88),
                _hit("car-phone", "手机", "手机选购对话", 0.84),
            ]
        ]


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return self.scores


def test_milvus_retriever_embeds_query_and_deduplicates_sources() -> None:
    client = FakeMilvusClient()
    embedder = FakeEmbedder()
    retriever = KnowledgeRetriever(client, embedder)

    results = retriever.search("我想买苹果，有什么区别？", limit=2)

    assert embedder.queries == ["我想买苹果，有什么区别？"]
    assert [result.source_id for result in results] == ["car-apple", "car-phone"]
    assert results[0].content == "苹果选购对话"
    assert results[0].revision == "3fe3a4223357"
    assert results[0].source_url == "https://github.com/ygan/Chinese-Ambiguous-Reference"
    assert results[0].license == "MIT"
    assert client.search_calls[0]["collection_name"] == "retail_knowledge"
    assert client.search_calls[0]["limit"] == 8


def test_empty_query_does_not_search_milvus() -> None:
    client = FakeMilvusClient()
    embedder = FakeEmbedder()
    retriever = KnowledgeRetriever(client, embedder)

    assert retriever.search("   ") == []
    assert embedder.queries == []
    assert client.search_calls == []


def test_results_below_relevance_threshold_are_removed() -> None:
    retriever = KnowledgeRetriever(
        FakeMilvusClient(),
        FakeEmbedder(),
        min_score=0.9,
    )

    results = retriever.search("我想买苹果")

    assert [result.source_id for result in results] == ["car-apple"]


def test_invalid_relevance_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        KnowledgeRetriever(FakeMilvusClient(), FakeEmbedder(), min_score=1.1)


def test_hybrid_retrieval_promotes_exact_bm25_match() -> None:
    lexical_retriever = BM25Retriever(
        [
            _bm25_document("car-apple", "苹果"),
            _bm25_document("car-phone", "手机"),
        ]
    )
    retriever = KnowledgeRetriever(
        FakeMilvusClient(),
        FakeEmbedder(),
        lexical_retriever=lexical_retriever,
    )

    results = retriever.search("我想买手机", limit=2)

    assert [result.source_id for result in results] == ["car-phone", "car-apple"]


def test_reranker_reorders_hybrid_candidates() -> None:
    lexical_retriever = BM25Retriever(
        [
            _bm25_document("car-apple", "苹果"),
            _bm25_document("car-phone", "手机"),
        ]
    )
    reranker = FakeReranker([0.1, 0.9])
    retriever = KnowledgeRetriever(
        FakeMilvusClient(),
        FakeEmbedder(),
        lexical_retriever=lexical_retriever,
        reranker=reranker,
        rerank_candidates=2,
    )

    results = retriever.search("我想买苹果", limit=2)

    assert [result.source_id for result in results] == ["car-phone", "car-apple"]
    assert [result.score for result in results] == [0.9, 0.1]
    assert reranker.calls[0][0] == "我想买苹果"
    assert reranker.calls[0][1][0].startswith("购物澄清案例：苹果\n")


def test_reranker_must_return_one_score_per_candidate() -> None:
    retriever = KnowledgeRetriever(
        FakeMilvusClient(),
        FakeEmbedder(),
        reranker=FakeReranker([0.9]),
        rerank_candidates=2,
    )

    with pytest.raises(ValueError, match="one score per candidate"):
        retriever.search("我想买苹果", limit=2)


def test_retrieve_candidates_skips_reranker() -> None:
    reranker = FakeReranker([0.9, 0.1])
    retriever = KnowledgeRetriever(
        FakeMilvusClient(),
        FakeEmbedder(),
        reranker=reranker,
    )

    candidates = retriever.retrieve_candidates("我想买苹果", limit=2)

    assert [candidate.source_id for candidate in candidates] == ["car-apple", "car-phone"]
    assert reranker.calls == []


def _hit(source_id: str, merchandise: str, text: str, score: float) -> dict:
    return {
        "distance": score,
        "entity": {
            "source_id": source_id,
            "text": text,
            "title": f"购物澄清案例：{merchandise}",
            "category": "购物歧义与澄清",
            "revision": "3fe3a4223357",
            "source_date": "2024-04-21",
            "source_url": "https://github.com/ygan/Chinese-Ambiguous-Reference",
            "license": "MIT",
        },
    }


def _bm25_document(source_id: str, merchandise: str) -> BM25Document:
    return BM25Document(
        source_id=source_id,
        title=f"购物澄清案例：{merchandise}",
        category="购物歧义与澄清",
        content=f"{merchandise}选购案例",
        revision="3fe3a4223357",
        source_date="2024-04-21",
        source_url="https://github.com/ygan/Chinese-Ambiguous-Reference",
        license="MIT",
        index_text=merchandise,
        product_phrases=(merchandise.casefold(),),
    )
