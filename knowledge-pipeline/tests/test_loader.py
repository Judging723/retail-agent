import json
from pathlib import Path

import pytest

from knowledge_pipeline.ingestion.loader import load_documents
from knowledge_pipeline.ingestion.schemas import KnowledgeDocument

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_knowledge.json"
PUBLIC_KNOWLEDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "customer-service-app"
    / "backend"
    / "data"
    / "knowledge.json"
)


def test_load_documents() -> None:
    documents = load_documents(FIXTURE_PATH)

    assert len(documents) == 2
    assert all(isinstance(document, KnowledgeDocument) for document in documents)
    assert documents[0].source_id == "return-policy"
    assert documents[0].keywords == ("退货", "七天", "签收")


def test_public_knowledge_has_expected_provenance() -> None:
    payload = json.loads(PUBLIC_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    documents = load_documents(PUBLIC_KNOWLEDGE_PATH)

    assert payload["dataset_name"] == "Chinese Ambiguous Reference"
    assert payload["raw_record_count"] == 1104
    assert payload["document_count"] == 1073
    assert len(documents) == 1073
    assert len({document.source_id for document in documents}) == 1073
    assert {document.license for document in documents} == {"MIT"}
    assert {document.revision for document in documents} == {"3fe3a4223357"}
    assert {document.source_url for document in documents} == {
        "https://github.com/ygan/Chinese-Ambiguous-Reference"
    }


def test_missing_documents_list_is_rejected(tmp_path: Path) -> None:
    path = _write_json(tmp_path, {"dataset_name": "invalid"})

    with pytest.raises(ValueError, match="'documents' list"):
        load_documents(path)


def test_duplicate_source_id_is_rejected(tmp_path: Path) -> None:
    source = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    source["documents"][1]["source_id"] = source["documents"][0]["source_id"]
    path = _write_json(tmp_path, source)

    with pytest.raises(ValueError, match="duplicate source_id: return-policy"):
        load_documents(path)


def test_empty_required_string_is_rejected(tmp_path: Path) -> None:
    source = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    source["documents"][0]["title"] = "   "
    path = _write_json(tmp_path, source)

    with pytest.raises(ValueError, match=r"documents\[0\]\.title"):
        load_documents(path)


def _write_json(tmp_path: Path, content: object) -> Path:
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return path
