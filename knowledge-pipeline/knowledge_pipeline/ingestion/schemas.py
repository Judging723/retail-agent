from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """A validated source document before retrieval-oriented splitting."""

    source_id: str
    title: str
    category: str
    content: str
    keywords: tuple[str, ...]
    revision: str
    source_date: str
    source_url: str
    license: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A deterministic retrieval unit derived from a source document."""

    chunk_id: str
    source_id: str
    text: str
    position: int
    title: str
    category: str
    keywords: tuple[str, ...]
    revision: str
    source_date: str
    source_url: str
    license: str
