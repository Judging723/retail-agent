import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge.json"
_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_CHINESE_SEQUENCE = re.compile(r"[\u3400-\u9fff]+")


@dataclass(frozen=True, slots=True)
class BM25Document:
    source_id: str
    title: str
    category: str
    content: str
    revision: str
    source_date: str
    source_url: str
    license: str
    index_text: str
    product_phrases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    document: BM25Document
    score: float


class BM25Retriever:
    """Small in-memory BM25 index for exact product and keyword matching."""

    def __init__(
        self,
        documents: Sequence[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self._documents = tuple(documents)
        self._k1 = k1
        self._b = b
        self._term_frequencies = tuple(
            Counter(tokenize_for_bm25(document.index_text)) for document in self._documents
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = Counter(
            token for frequencies in self._term_frequencies for token in frequencies
        )

    @classmethod
    def from_json(cls, path: Path = DEFAULT_KNOWLEDGE_PATH) -> "BM25Retriever":
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            BM25Document(
                source_id=item["source_id"],
                title=item["title"],
                category=item["category"],
                content=item["content"],
                revision=item["revision"],
                source_date=item["source_date"],
                source_url=item["source_url"],
                license=item["license"],
                index_text=_build_index_text(item["title"], item["keywords"]),
                product_phrases=_extract_product_phrases(item["title"]),
            )
            for item in payload["documents"]
        ]
        return cls(documents)

    def search(self, query: str, *, limit: int = 8) -> list[BM25SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        normalized_query = _normalize(query)
        query_tokens = set(tokenize_for_bm25(normalized_query))
        if not query_tokens or not self._documents:
            return []

        results: list[BM25SearchResult] = []
        for document, frequencies, document_length in zip(
            self._documents,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            if not any(phrase in normalized_query for phrase in document.product_phrases):
                continue

            score = sum(
                self._term_score(
                    token,
                    term_frequency=frequencies[token],
                    document_length=document_length,
                )
                for token in query_tokens
                if frequencies[token] > 0
            )
            if score > 0:
                results.append(BM25SearchResult(document=document, score=score))

        results.sort(key=lambda result: (-result.score, result.document.source_id))
        return results[:limit]

    def _term_score(
        self,
        token: str,
        *,
        term_frequency: int,
        document_length: int,
    ) -> float:
        document_count = len(self._documents)
        document_frequency = self._document_frequencies[token]
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_ratio = document_length / self._average_length if self._average_length else 0.0
        denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
        return inverse_document_frequency * term_frequency * (self._k1 + 1) / denominator


def tokenize_for_bm25(text: str) -> tuple[str, ...]:
    normalized = _normalize(text)
    tokens = _LATIN_OR_NUMBER.findall(normalized)

    for sequence in _CHINESE_SEQUENCE.findall(normalized):
        if len(sequence) == 1:
            tokens.append(sequence)
        else:
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))

    return tuple(tokens)


def _build_index_text(title: str, keywords: list[str]) -> str:
    specific_title = title.split("：", maxsplit=1)[-1]
    specific_keywords = [keyword for keyword in keywords if keyword not in {"购物咨询", "澄清"}]
    return " ".join([specific_title, *specific_keywords])


def _extract_product_phrases(title: str) -> tuple[str, ...]:
    specific_title = _normalize(title.split("：", maxsplit=1)[-1])
    return tuple(
        phrase.strip()
        for phrase in re.split(r"[、,，/]", specific_title)
        if len(phrase.strip()) >= 2
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()
