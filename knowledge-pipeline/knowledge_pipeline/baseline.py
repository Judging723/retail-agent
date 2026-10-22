from collections.abc import Sequence

from knowledge_pipeline.schemas import KnowledgeDocument


def keyword_search(
    documents: Sequence[KnowledgeDocument],
    query: str,
    *,
    limit: int = 3,
) -> list[str]:
    """Return source IDs ranked by exact keyword matches in the query."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    normalized_query = query.casefold()
    scored_documents: list[tuple[int, str]] = []

    for document in documents:
        score = sum(
            len(keyword) for keyword in document.keywords if keyword.casefold() in normalized_query
        )
        if score > 0:
            scored_documents.append((score, document.source_id))

    scored_documents.sort(key=lambda item: (-item[0], item[1]))
    return [source_id for _, source_id in scored_documents[:limit]]
