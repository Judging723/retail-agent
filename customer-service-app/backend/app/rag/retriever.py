from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MILVUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "knowledge-pipeline"
    / "data"
    / "vector_store"
    / "retail_agent.db"
)
DEFAULT_COLLECTION_NAME = "retail_knowledge"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

OUTPUT_FIELDS = [
    "source_id",
    "text",
    "title",
    "category",
    "revision",
    "source_date",
    "source_url",
    "license",
]


class QueryEmbedder(Protocol):
    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerQueryEmbedder:
    """Encode user questions with the same pinned model used by the offline pipeline."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            DEFAULT_MODEL_NAME,
            revision=DEFAULT_MODEL_REVISION,
        )

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self._model.encode(
            [f"{QUERY_INSTRUCTION}{text}" for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


@dataclass(frozen=True)
class RetrievedDocument:
    source_id: str
    title: str
    category: str
    content: str
    revision: str
    source_date: str
    source_url: str
    license: str
    score: float


class KnowledgeRetriever:
    """Search the Milvus collection built by the offline knowledge pipeline."""

    def __init__(
        self,
        client: Any,
        embedder: QueryEmbedder,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name

    @classmethod
    def from_milvus(
        cls,
        uri: str = str(DEFAULT_MILVUS_PATH),
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> "KnowledgeRetriever":
        from pymilvus import MilvusClient

        embedder = SentenceTransformerQueryEmbedder()
        client = MilvusClient(uri=uri)
        client.load_collection(collection_name=collection_name)
        return cls(
            client,
            embedder,
            collection_name=collection_name,
        )

    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        if not query.strip():
            return []
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_embedding = self._embedder.encode_queries([query])[0]
        response = self._client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            limit=limit * 4,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )

        results: list[RetrievedDocument] = []
        seen_source_ids: set[str] = set()
        for hit in response[0]:
            entity = hit["entity"]
            source_id = entity["source_id"]
            if source_id in seen_source_ids:
                continue

            seen_source_ids.add(source_id)
            results.append(
                RetrievedDocument(
                    source_id=source_id,
                    title=entity["title"],
                    category=entity["category"],
                    content=entity["text"],
                    revision=entity["revision"],
                    source_date=entity["source_date"],
                    source_url=entity["source_url"],
                    license=entity["license"],
                    score=float(hit["distance"]),
                )
            )
            if len(results) == limit:
                break

        return results
