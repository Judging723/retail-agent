import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_registry.json"
ALLOWED_REVIEW_STATUSES = {
    "approved_for_experiment",
    "approved_with_attribution",
    "approval_required",
    "license_review_required",
}


def test_dataset_registry_has_unique_ids_and_valid_review_statuses() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    datasets = registry["datasets"]
    dataset_ids = [dataset["id"] for dataset in datasets]

    assert len(dataset_ids) == len(set(dataset_ids))
    assert all(dataset["review_status"] in ALLOWED_REVIEW_STATUSES for dataset in datasets)


def test_approved_datasets_have_a_verified_license() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    approved = [
        dataset
        for dataset in registry["datasets"]
        if dataset["review_status"].startswith("approved")
    ]

    assert approved
    assert all(dataset["license"] and dataset["license_url"] for dataset in approved)
    assert all(dataset["rag_knowledge_base_candidate"] is False for dataset in approved)
