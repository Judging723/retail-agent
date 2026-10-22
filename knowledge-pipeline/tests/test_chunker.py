from itertools import pairwise

import pytest

from knowledge_pipeline.chunker import chunk_document
from knowledge_pipeline.schemas import KnowledgeDocument


def test_short_document_creates_one_chunk() -> None:
    document = _make_document("短文本")

    chunks = chunk_document(document, max_chars=10, overlap_chars=3)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "shipping-policy:1.0:0000"
    assert chunks[0].text == "短文本"
    assert chunks[0].position == 0
    assert chunks[0].category == document.category
    assert chunks[0].keywords == document.keywords


def test_long_document_creates_overlapping_chunks() -> None:
    document = _make_document("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    chunks = chunk_document(document, max_chars=10, overlap_chars=3)

    assert [chunk.text for chunk in chunks] == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRSTUVWX",
        "VWXYZ",
    ]
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert all(previous.text[-3:] == current.text[:3] for previous, current in pairwise(chunks))
    assert [chunk.chunk_id for chunk in chunks] == [
        "shipping-policy:1.0:0000",
        "shipping-policy:1.0:0001",
        "shipping-policy:1.0:0002",
        "shipping-policy:1.0:0003",
    ]


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_settings_are_rejected(max_chars: int, overlap_chars: int) -> None:
    document = _make_document("测试文本")

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars)


def _make_document(content: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        source_id="shipping-policy",
        title="物流规则",
        category="物流",
        content=content,
        keywords=("物流", "发货"),
        version="1.0",
        effective_date="2026-10-01",
    )
