from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.rag.bm25 import DEFAULT_KNOWLEDGE_PATH, BM25Retriever, BM25SearchResult
from app.rag.reranker import PairReranker

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
DEFAULT_RELEVANCE_THRESHOLD = 0.52
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
        min_score: float = DEFAULT_RELEVANCE_THRESHOLD,
        lexical_retriever: BM25Retriever | None = None,
        reranker: PairReranker | None = None,
        rerank_candidates: int = 8,
        rrf_k: int = 60,
    ) -> None:
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be greater than zero")

        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name
        self._min_score = min_score
        self._lexical_retriever = lexical_retriever
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._rrf_k = rrf_k

    @classmethod
    def from_milvus(
        cls,
        uri: str = str(DEFAULT_MILVUS_PATH),
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        min_score: float = DEFAULT_RELEVANCE_THRESHOLD,
        knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
        reranker: PairReranker | None = None,
        rerank_candidates: int = 8,
    ) -> "KnowledgeRetriever":
        from pymilvus import MilvusClient

        embedder = SentenceTransformerQueryEmbedder()
        client = MilvusClient(uri=uri)
        client.load_collection(collection_name=collection_name)
        return cls(
            client,
            embedder,
            collection_name=collection_name,
            min_score=min_score,
            lexical_retriever=BM25Retriever.from_json(knowledge_path),
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )

    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        if not query.strip():
            return []
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if self._lexical_retriever is None and self._reranker is None:
            return self._search_vectors(query, limit=limit)

        if self._reranker is None:
            candidate_limit = max(8, limit * 4)
        else:
            candidate_limit = max(self._rerank_candidates, limit)

        candidates = self.retrieve_candidates(query, limit=candidate_limit)
        if self._reranker is None:
            return candidates[:limit]
        return self._rerank(query, candidates, limit=limit)

    def retrieve_candidates(
        self,
        query: str,
        *,
        limit: int = 8,
    ) -> list[RetrievedDocument]:
        """Return the hybrid retrieval candidates before CrossEncoder reranking."""

        if not query.strip():
            return []
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if self._lexical_retriever is None:
            return self._search_vectors(query, limit=limit)

        vector_results = self._search_vectors(query, limit=limit)
        lexical_results = self._lexical_retriever.search(query, limit=limit)
        return self._fuse_results(vector_results, lexical_results, limit=limit)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _search_vectors(self, query: str, *, limit: int) -> list[RetrievedDocument]:
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
            score = float(hit["distance"])
            if score < self._min_score:
                continue

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
                    score=score,
                )
            )
            if len(results) == limit:
                break

        return results

    def _rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedDocument],
        *,
        limit: int,
    ) -> list[RetrievedDocument]:
        if self._reranker is None:
            return list(candidates[:limit])

        passages = [f"{document.title}\n{document.content}" for document in candidates]
        scores = self._reranker.score(query, passages)
        if len(scores) != len(candidates):
            raise ValueError("reranker must return one score per candidate")

        ranked = sorted(
            enumerate(zip(candidates, scores, strict=True)),
            key=lambda item: (-item[1][1], item[0]),
        )
        return [_with_score(document, score) for _, (document, score) in ranked[:limit]]

    def _fuse_results(
        self,
        vector_results: Sequence[RetrievedDocument],
        lexical_results: Sequence[BM25SearchResult],
        *,
        limit: int,
    ) -> list[RetrievedDocument]:
        documents: dict[str, RetrievedDocument] = {}
        fused_scores: Counter[str] = Counter()
        retrieval_channels: dict[str, set[str]] = {}

        for rank, document in enumerate(vector_results, start=1):
            documents[document.source_id] = document
            fused_scores[document.source_id] += 1 / (self._rrf_k + rank)
            retrieval_channels.setdefault(document.source_id, set()).add("vector")

        for rank, result in enumerate(lexical_results, start=1):
            source_id = result.document.source_id
            documents.setdefault(source_id, _from_bm25_result(result))
            fused_scores[source_id] += 1 / (self._rrf_k + rank)
            retrieval_channels.setdefault(source_id, set()).add("bm25")

        ranked_source_ids = sorted(
            documents,
            key=lambda source_id: (
                -fused_scores[source_id],
                -len(retrieval_channels[source_id]),
                source_id,
            ),
        )
        return [
            _with_score(documents[source_id], fused_scores[source_id])
            for source_id in ranked_source_ids[:limit]
        ]


def _from_bm25_result(result: BM25SearchResult) -> RetrievedDocument:
    document = result.document
    return RetrievedDocument(
        source_id=document.source_id,
        title=document.title,
        category=document.category,
        content=document.content,
        revision=document.revision,
        source_date=document.source_date,
        source_url=document.source_url,
        license=document.license,
        score=result.score,
    )


def _with_score(document: RetrievedDocument, score: float) -> RetrievedDocument:
    return RetrievedDocument(
        source_id=document.source_id,
        title=document.title,
        category=document.category,
        content=document.content,
        revision=document.revision,
        source_date=document.source_date,
        source_url=document.source_url,
        license=document.license,
        score=score,
    )
