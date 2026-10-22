import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge.json"


@dataclass(frozen=True)
class KnowledgeDocument:
    source_id: str
    title: str
    category: str
    content: str
    keywords: tuple[str, ...]
    version: str
    effective_date: str


@dataclass(frozen=True)
class RetrievedDocument:
    source_id: str
    title: str
    category: str
    content: str
    version: str
    effective_date: str
    score: int


class KnowledgeRetriever:
    def __init__(self, documents: Sequence[KnowledgeDocument]) -> None:
        self._documents = tuple(documents)

    @classmethod
    def from_json(cls, path: Path = DEFAULT_KNOWLEDGE_PATH) -> "KnowledgeRetriever":
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            KnowledgeDocument(
                source_id=item["source_id"],
                title=item["title"],
                category=item["category"],
                content=item["content"],
                keywords=tuple(item["keywords"]),
                version=item["version"],
                effective_date=item["effective_date"],
            )
            for item in payload["documents"]
        ]
        return cls(documents)

    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        normalized_query = query.casefold()
        results: list[RetrievedDocument] = []

        for document in self._documents:
            matched_keywords = [
                keyword for keyword in document.keywords if keyword.casefold() in normalized_query
            ]
            if not matched_keywords:
                continue

            results.append(
                RetrievedDocument(
                    source_id=document.source_id,
                    title=document.title,
                    category=document.category,
                    content=document.content,
                    version=document.version,
                    effective_date=document.effective_date,
                    score=sum(len(keyword) for keyword in matched_keywords),
                )
            )

        results.sort(key=lambda document: (-document.score, document.source_id))
        return results[:limit]
