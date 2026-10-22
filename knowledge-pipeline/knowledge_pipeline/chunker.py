from knowledge_pipeline.schemas import KnowledgeChunk, KnowledgeDocument


def chunk_document(
    document: KnowledgeDocument,
    max_chars: int = 300,
    overlap_chars: int = 50,
) -> list[KnowledgeChunk]:
    """Split one document into deterministic fixed-width character chunks."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[KnowledgeChunk] = []
    start = 0
    position = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document.source_id}:{document.version}:{position:04d}",
                source_id=document.source_id,
                text=text[start:end],
                position=position,
                title=document.title,
                category=document.category,
                keywords=document.keywords,
                version=document.version,
                effective_date=document.effective_date,
            )
        )

        if end == len(text):
            break

        start = end - overlap_chars
        position += 1

    return chunks
