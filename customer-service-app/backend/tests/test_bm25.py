from app.rag import BM25Document, BM25Retriever
from app.rag.bm25 import DEFAULT_KNOWLEDGE_PATH, tokenize_for_bm25


def test_chinese_product_name_can_be_retrieved() -> None:
    retriever = BM25Retriever(
        [
            _document("apple", "苹果", "苹果 红富士"),
            _document("sun-shirt", "防晒衣", "防晒衣 UPF50"),
        ]
    )

    results = retriever.search("有没有适合夏天的防晒衣？")

    assert [result.document.source_id for result in results] == ["sun-shirt"]
    assert results[0].score > 0


def test_unrelated_query_returns_no_lexical_results() -> None:
    retriever = BM25Retriever([_document("apple", "苹果", "苹果 红富士")])

    assert retriever.search("帮我用 Python 写快速排序") == []


def test_partial_word_does_not_impersonate_a_product_name() -> None:
    retriever = BM25Retriever([_document("wall-sticker", "黑板装饰墙贴", "黑板装饰墙贴")])

    assert retriever.search("Python 的装饰器怎么写？") == []


def test_tokenizer_keeps_latin_terms_and_chinese_bigrams() -> None:
    tokens = tokenize_for_bm25("iPhone 15 防晒衣")

    assert "iphone" in tokens
    assert "15" in tokens
    assert "防晒" in tokens
    assert "晒衣" in tokens


def test_public_knowledge_exact_product_is_ranked_first() -> None:
    retriever = BM25Retriever.from_json(DEFAULT_KNOWLEDGE_PATH)

    results = retriever.search("水杨酸棉片怎么选", limit=2)

    assert results[0].document.source_id == "car-e1aa19424d72a1de"
    assert results[0].document.title == "购物澄清案例：水杨酸棉片"


def _document(source_id: str, title: str, index_text: str) -> BM25Document:
    return BM25Document(
        source_id=source_id,
        title=title,
        category="购物歧义与澄清",
        content=f"{title}的澄清案例",
        revision="test-revision",
        source_date="2024-04-21",
        source_url="https://example.com/dataset",
        license="MIT",
        index_text=index_text,
        product_phrases=(title.casefold(),),
    )
