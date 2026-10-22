from app.rag import KnowledgeRetriever


def test_demo_knowledge_base_can_be_loaded_and_searched() -> None:
    retriever = KnowledgeRetriever.from_json()

    results = retriever.search("商品刚买就降价了，可以申请价保吗？")

    assert [result.source_id for result in results] == ["price-protection"]
    assert results[0].category == "优惠"
    assert results[0].version == "1.0"
    assert results[0].effective_date == "2026-10-01"


def test_demo_knowledge_base_returns_empty_for_unrelated_question() -> None:
    retriever = KnowledgeRetriever.from_json()

    assert retriever.search("今天的天气怎么样？") == []
