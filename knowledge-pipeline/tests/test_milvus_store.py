import pytest

pymilvus = pytest.importorskip("pymilvus")

from knowledge_pipeline.ingestion.schemas import KnowledgeChunk
from knowledge_pipeline.retrieval.milvus_store import MilvusVectorStore


def test_chunks_can_be_stored_and_searched(tmp_path) -> None:
    database_path = str(tmp_path / "test.db")
    client = pymilvus.MilvusClient(uri=database_path)
    store = MilvusVectorStore(client, collection_name="test_knowledge")
    chunks = [
        _make_chunk("return-policy:1.0:0000", "return-policy", "七天无理由退货"),
        _make_chunk("shipping-policy:1.0:0000", "shipping-policy", "订单将在两天内发货"),
    ]

    store.replace_collection(vector_size=2)
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])
    client.close()

    client = pymilvus.MilvusClient(uri=database_path)
    store = MilvusVectorStore(client, collection_name="test_knowledge")
    store.load()
    results = store.search([0.9, 0.1], limit=2)

    assert [result.source_id for result in results] == ["return-policy", "shipping-policy"]
    assert results[0].chunk_id == "return-policy:1.0:0000"
    assert results[0].text == "七天无理由退货"
    assert results[0].score > results[1].score
    client.close()


def test_upsert_rejects_mismatched_input_lengths(tmp_path) -> None:
    client = pymilvus.MilvusClient(uri=str(tmp_path / "test.db"))
    store = MilvusVectorStore(client, collection_name="test_knowledge")
    store.replace_collection(vector_size=2)

    with pytest.raises(
        ValueError,
        match="chunks and embeddings must have the same length",
    ):
        store.upsert([_make_chunk("one", "source", "内容")], [])

    client.close()


def _make_chunk(chunk_id: str, source_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        position=0,
        title="测试知识",
        category="test",
        keywords=(),
        revision="test-revision",
        source_date="2024-04-21",
        source_url="https://example.com/dataset",
        license="MIT",
    )
