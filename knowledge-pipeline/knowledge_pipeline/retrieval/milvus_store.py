from collections.abc import Sequence
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

from knowledge_pipeline.ingestion.schemas import KnowledgeChunk

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


@dataclass(frozen=True, slots=True)
class MilvusSearchResult:
    chunk_id: str
    source_id: str
    text: str
    title: str
    category: str
    revision: str
    source_date: str
    source_url: str
    license: str
    score: float


class MilvusVectorStore:
    """Persist and search knowledge chunks through the MilvusClient API."""

    def __init__(self, client: MilvusClient, collection_name: str) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")

        self._client = client
        self.collection_name = collection_name.strip()

    def replace_collection(self, vector_size: int) -> None:
        """Replace the collection when rebuilding the complete knowledge index."""

        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=vector_size,
        )
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="revision", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_date", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="source_url", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="license", datatype=DataType.VARCHAR, max_length=64)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        data = [
            {
                "id": chunk.chunk_id,
                "vector": list(embedding),
                "source_id": chunk.source_id,
                "text": chunk.text,
                "title": chunk.title,
                "category": chunk.category,
                "revision": chunk.revision,
                "source_date": chunk.source_date,
                "source_url": chunk.source_url,
                "license": chunk.license,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        if data:
            self._client.upsert(collection_name=self.collection_name, data=data)
            self._client.flush(collection_name=self.collection_name)

    def load(self) -> None:
        """Load an existing collection before serving search requests."""

        self._client.load_collection(collection_name=self.collection_name)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 3,
    ) -> list[MilvusSearchResult]:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        response = self._client.search(
            collection_name=self.collection_name,
            data=[list(query_embedding)],
            limit=limit,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )

        return [
            MilvusSearchResult(
                chunk_id=_result_string(hit, "id"),
                source_id=_entity_string(hit, "source_id"),
                text=_entity_string(hit, "text"),
                title=_entity_string(hit, "title"),
                category=_entity_string(hit, "category"),
                revision=_entity_string(hit, "revision"),
                source_date=_entity_string(hit, "source_date"),
                source_url=_entity_string(hit, "source_url"),
                license=_entity_string(hit, "license"),
                score=float(hit["distance"]),
            )
            for hit in response[0]
        ]


def _result_string(hit: dict, field: str) -> str:
    value = hit.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Milvus search result is missing string field: {field}")
    return value


def _entity_string(hit: dict, field: str) -> str:
    entity = hit.get("entity")
    if not isinstance(entity, dict):
        raise ValueError("Milvus search result is missing entity metadata")
    return _result_string(entity, field)
