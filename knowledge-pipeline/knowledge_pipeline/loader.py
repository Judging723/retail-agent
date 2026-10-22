import json
from pathlib import Path
from typing import Any

from knowledge_pipeline.schemas import KnowledgeDocument

_STRING_FIELDS = (
    "source_id",
    "title",
    "category",
    "content",
    "version",
    "effective_date",
)


def load_documents(path: Path) -> list[KnowledgeDocument]:
    """Load and validate knowledge documents from a UTF-8 JSON file."""

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError("knowledge file must contain a JSON object")

    raw_documents = raw_data.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("knowledge file must contain a 'documents' list")

    documents: list[KnowledgeDocument] = []
    seen_source_ids: set[str] = set()

    for index, raw_document in enumerate(raw_documents):
        document = _parse_document(raw_document, index=index)
        if document.source_id in seen_source_ids:
            raise ValueError(f"duplicate source_id: {document.source_id}")

        seen_source_ids.add(document.source_id)
        documents.append(document)

    return documents


def _parse_document(raw_document: Any, *, index: int) -> KnowledgeDocument:
    if not isinstance(raw_document, dict):
        raise ValueError(f"documents[{index}] must be a JSON object")

    values = {
        field: _required_string(raw_document, field=field, index=index) for field in _STRING_FIELDS
    }

    raw_keywords = raw_document.get("keywords")
    if not isinstance(raw_keywords, list):
        raise ValueError(f"documents[{index}].keywords must be a list")

    keywords: list[str] = []
    for keyword_index, keyword in enumerate(raw_keywords):
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError(
                f"documents[{index}].keywords[{keyword_index}] must be a non-empty string"
            )
        keywords.append(keyword.strip())

    return KnowledgeDocument(
        source_id=values["source_id"],
        title=values["title"],
        category=values["category"],
        content=values["content"],
        keywords=tuple(keywords),
        version=values["version"],
        effective_date=values["effective_date"],
    )


def _required_string(raw_document: dict[str, Any], *, field: str, index: int) -> str:
    value = raw_document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"documents[{index}].{field} must be a non-empty string")
    return value.strip()
