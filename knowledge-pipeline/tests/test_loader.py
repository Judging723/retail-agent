import json
from pathlib import Path

import pytest

from knowledge_pipeline.loader import load_documents
from knowledge_pipeline.schemas import KnowledgeDocument

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_knowledge.json"


def test_load_documents() -> None:
    documents = load_documents(FIXTURE_PATH)

    assert len(documents) == 2
    assert all(isinstance(document, KnowledgeDocument) for document in documents)
    assert documents[0].source_id == "return-policy"
    assert documents[0].keywords == ("退货", "七天", "签收")


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
