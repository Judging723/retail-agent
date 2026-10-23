import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from model_pipeline.preparation import Split, assign_split

SOURCE_NAME = "Kapibala Sales Dialogues"
SOURCE_URL = "https://huggingface.co/datasets/kapibala-ai/kapibala-sales-dialogues"
SOURCE_REVISION = "c94eb4fbc32d6f73f15bdb3dfc84600afd809844"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
META_SHA256 = "6e838c0f17787cbba7a4385dfb64e5d0444005b9f2c9c0691036bb46f9879099"
CONVERSATIONS_L3_SHA256 = "be7460761e47b05725907a658e68814f5afca169fe501c3c52adc8ef3fd64459"
META_URL = f"{SOURCE_URL}/resolve/{SOURCE_REVISION}/data/meta.jsonl"
CONVERSATIONS_L3_URL = (
    f"{SOURCE_URL}/resolve/{SOURCE_REVISION}/data/with_text/conversations_L3.jsonl"
)
LABELS = ("negative", "neutral", "positive")


def prepare_sentiment_dataset(
    meta_path: Path,
    conversations_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Prepare Chinese customer messages with the source model's sentiment pseudo-labels."""

    meta_bytes = _read_verified(meta_path, META_SHA256)
    conversation_bytes = _read_verified(conversations_path, CONVERSATIONS_L3_SHA256)
    metadata = _load_unique_jsonl(meta_bytes, "metadata")
    conversations = _load_unique_jsonl(conversation_bytes, "conversations")

    missing_meta = sorted(set(conversations) - set(metadata))
    if missing_meta:
        raise ValueError(f"conversations missing metadata: {missing_meta[:3]}")

    candidates: list[dict[str, Any]] = []
    conversations_used: set[str] = set()
    setting_ids: set[str] = set()
    for conversation_id in sorted(conversations):
        meta = metadata[conversation_id]
        if meta.get("language") != "zh":
            continue
        setting_id = _required_string(meta, "setting_id", conversation_id)
        messages = conversations[conversation_id].get("messages")
        if not isinstance(messages, list):
            raise TypeError(f"{conversation_id}: messages must be a list")

        for turn, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"{conversation_id}:{turn}: message must be an object")
            if message.get("role") != "b":
                continue
            text = _normalize_text(message.get("content"), conversation_id, turn)
            probabilities = _sentiment_probabilities(message.get("L3"), conversation_id, turn)
            ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
            label = ranked[0][0]
            margin = ranked[0][1] - ranked[1][1]
            conversations_used.add(conversation_id)
            setting_ids.add(setting_id)
            candidates.append(
                {
                    "id": f"{conversation_id}:turn-{turn:02d}",
                    "text": text,
                    "label": label,
                    "label_id": LABELS.index(label),
                    "probabilities": probabilities,
                    "teacher_confidence": ranked[0][1],
                    "teacher_margin": margin,
                    "metadata": {
                        "conversation_id": conversation_id,
                        "turn": turn,
                        "setting_id": setting_id,
                        "domain": meta.get("domain"),
                        "source": meta.get("source"),
                        "source_url": SOURCE_URL,
                        "source_revision": SOURCE_REVISION,
                        "license": LICENSE,
                    },
                }
            )

    records, duplicates_removed = _deduplicate_by_text(candidates)
    by_split: dict[Split, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    groups_by_split: dict[Split, set[str]] = {"train": set(), "dev": set(), "test": set()}
    labels_by_split: dict[Split, Counter[str]] = {
        "train": Counter(),
        "dev": Counter(),
        "test": Counter(),
    }
    for record in records:
        setting_id = record["metadata"]["setting_id"]
        split = assign_split(f"kapibala:{setting_id}")
        by_split[split].append(record)
        groups_by_split[split].add(setting_id)
        labels_by_split[split][record["label"]] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        path = output_dir / f"{split}.jsonl"
        serialized = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in by_split[split]
        ).encode("utf-8")
        path.write_bytes(serialized)
        outputs[split] = {
            "file": path.name,
            "records": len(by_split[split]),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        }

    label_counts = Counter(record["label"] for record in records)
    return {
        "format_version": 1,
        "task": "three-class Chinese customer-message sentiment classification",
        "dataset": {
            "name": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "raw_files": {
                meta_path.name: META_SHA256,
                conversations_path.name: CONVERSATIONS_L3_SHA256,
            },
        },
        "label_policy": {
            "classes": list(LABELS),
            "rule": "argmax(p_negative, p_neutral, p_positive)",
            "teacher_model": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "warning": (
                "Labels are model-generated pseudo-labels on synthetic conversations, not human gold "
                "annotations. Metrics measure agreement with the teacher labels."
            ),
        },
        "transformation": {
            "language": "zh",
            "role": "b (customer) only",
            "split_strategy": (
                "sha256('kapibala:' + setting_id) bucket: train 0-79, dev 80-89, test 90-99"
            ),
            "deduplication": "exact normalized message text; conflicting labels are rejected",
            "pii_policy": "no redaction; the source dataset declares all conversations synthetic",
        },
        "statistics": {
            "raw_conversations": len(conversations),
            "chinese_conversations": len(conversations_used),
            "setting_groups": len(setting_ids),
            "candidate_messages": len(candidates),
            "duplicates_removed": duplicates_removed,
            "output_messages": len(records),
            "labels": {label: label_counts[label] for label in LABELS},
            "low_margin_messages_below_0_1": sum(
                record["teacher_margin"] < 0.1 for record in records
            ),
            "samples_by_split": {split: len(by_split[split]) for split in by_split},
            "groups_by_split": {split: len(groups_by_split[split]) for split in groups_by_split},
            "labels_by_split": {
                split: {label: labels_by_split[split][label] for label in LABELS}
                for split in labels_by_split
            },
        },
        "outputs": outputs,
    }


def _read_verified(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"raw dataset checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return data


def _load_unique_jsonl(data: bytes, description: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{description}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise TypeError(f"{description}:{line_number}: row must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{description}:{line_number}: id must be a non-empty string")
        if record_id in records:
            raise ValueError(f"{description}:{line_number}: duplicate id {record_id}")
        records[record_id] = record
    return records


def _sentiment_probabilities(
    raw_l3: Any,
    conversation_id: str,
    turn: int,
) -> dict[str, float]:
    if not isinstance(raw_l3, dict):
        raise TypeError(f"{conversation_id}:{turn}: L3 must be an object")
    probabilities: dict[str, float] = {}
    for label in LABELS:
        value = raw_l3.get(f"p_{label}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{conversation_id}:{turn}: p_{label} must be numeric")
        probability = float(value)
        if not 0 <= probability <= 1:
            raise ValueError(f"{conversation_id}:{turn}: p_{label} must be in [0, 1]")
        probabilities[label] = probability
    if abs(sum(probabilities.values()) - 1.0) > 0.01:
        raise ValueError(f"{conversation_id}:{turn}: sentiment probabilities must sum to 1")
    return probabilities


def _required_string(record: dict[str, Any], key: str, record_id: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{record_id}: {key} must be a non-empty string")
    return value


def _normalize_text(raw_text: Any, conversation_id: str, turn: int) -> str:
    if not isinstance(raw_text, str):
        raise TypeError(f"{conversation_id}:{turn}: content must be a string")
    text = " ".join(raw_text.split())
    if not text:
        raise ValueError(f"{conversation_id}:{turn}: content must not be empty")
    return text


def _deduplicate_by_text(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    for record in candidates:
        text = record["text"]
        previous = unique.get(text)
        if previous is not None and previous["label"] != record["label"]:
            raise ValueError(f"conflicting pseudo-labels for duplicate text: {text[:40]!r}")
        unique.setdefault(text, record)
    return list(unique.values()), len(candidates) - len(unique)
