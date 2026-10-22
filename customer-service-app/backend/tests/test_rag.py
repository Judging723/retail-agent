from collections.abc import Sequence

import pytest

from app.rag import KnowledgeRetriever


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
