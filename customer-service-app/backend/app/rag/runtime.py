import logging
from collections.abc import Callable
from threading import Lock
from time import perf_counter
from typing import Literal, Protocol

from app.rag.retriever import KnowledgeRetriever, RetrievedDocument

logger = logging.getLogger("app.rag")
RuntimeStatus = Literal["not_initialized", "ready", "degraded", "closed"]


class KnowledgeSearcher(Protocol):
    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]: ...


class KnowledgeRuntime:
    """Own the retriever lifecycle and keep RAG failures outside the chat critical path."""

    def __init__(self, factory: Callable[[], KnowledgeRetriever]) -> None:
        self._factory = factory
        self._retriever: KnowledgeRetriever | None = None
        self._status: RuntimeStatus = "not_initialized"
        self._lock = Lock()

    @property
    def status(self) -> RuntimeStatus:
        with self._lock:
            return self._status

    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        retriever = self._get_or_initialize()
        if retriever is None:
            return []

        try:
            return retriever.search(query, limit=limit)
        # RAG is auxiliary; any backend/model failure must fall back to a chat without context.
        except Exception as exc:  # noqa: BLE001
            self._degrade(retriever, exc, event="rag_search_failed")
            return []

    def warmup(self) -> bool:
        started_at = perf_counter()
        self.search("商品咨询", limit=1)
        ready = self.status == "ready"
        logger.info(
            "rag_warmup_finished status=%s duration_ms=%.2f",
            "ready" if ready else "degraded",
            (perf_counter() - started_at) * 1000,
        )
        return ready

    def close(self) -> None:
        with self._lock:
            retriever = self._retriever
            self._retriever = None
            self._status = "closed"

        if retriever is not None:
            retriever.close()

    def _get_or_initialize(self) -> KnowledgeRetriever | None:
        with self._lock:
            if self._status == "degraded":
                return None
            if self._retriever is not None:
                return self._retriever

            try:
                self._retriever = self._factory()
            except Exception as exc:
                self._status = "degraded"
                logger.exception(
                    "rag_initialization_failed error_type=%s",
                    type(exc).__name__,
                )
                return None

            self._status = "ready"
            return self._retriever

    def _degrade(
        self,
        retriever: KnowledgeRetriever,
        exc: Exception,
        *,
        event: str,
    ) -> None:
        with self._lock:
            if self._retriever is retriever:
                self._retriever = None
                self._status = "degraded"

        retriever.close()
        logger.exception("%s error_type=%s", event, type(exc).__name__)
