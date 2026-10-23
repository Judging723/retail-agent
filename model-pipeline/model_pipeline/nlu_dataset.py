import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_pipeline.preparation import (
    ASSISTANT_SPEAKERS,
    KNOWN_SPEAKERS,
    USER_SPEAKERS,
    Role,
    Split,
    assign_split,
)
from model_pipeline.redaction import redact_text

SOURCE_REVISION = "3fe3a42233571914ee4a5ccaec3f3af97966d9c9"
RAW_SHA256 = "36631734421c1b892b30e9f7298ab1f29bfc1de5ccc59fe14bbc91a1256377fe"
SOURCE_URL = "https://github.com/ygan/Chinese-Ambiguous-Reference"

# The pinned JSON and the repository's prose label table are not fully consistent.
# These sets therefore encode an explicit weak-label rule based on observed service-response
# behavior, rather than pretending that the customer utterances have direct binary labels.
CLARIFICATION_TRIGGER_CODES = frozenset({2, 3, 4, 6, 7})
NON_CLARIFICATION_CODES = frozenset({1, 5, 8})

_CANONICAL_FIELDS = (
    "date",
    "display_method",
    "location",
    "merchandise",
    "shop",
    "shopBackground",
    "total",
    "total_split",
    "split_sentence_type",
)
_SPEAKER_PATTERN = re.compile(
    rf"^({'|'.join(re.escape(item) for item in sorted(KNOWN_SPEAKERS, key=len, reverse=True))})"
    r"[：:]?[“\"]?(.*)$"
)


@dataclass
class LabeledTurn:
    role: Role
    text: str
    annotation_codes: list[int]


@dataclass
class NLUParseDiagnostics:
    continuation_lines: int = 0
    consecutive_turns_merged: int = 0


def parse_labeled_dialogue(
    raw_turns: list[Any], raw_labels: list[Any]
) -> tuple[list[LabeledTurn], NLUParseDiagnostics]:
    """Align noisy source utterances with their numeric annotations and canonical roles."""

    if len(raw_turns) != len(raw_labels):
        raise ValueError("utterance and annotation counts do not match")

    diagnostics = NLUParseDiagnostics()
    turns: list[LabeledTurn] = []
    for raw_turn, raw_label in zip(raw_turns, raw_labels, strict=True):
        if not isinstance(raw_turn, str) or not raw_turn.strip():
            raise ValueError("utterance must be a non-empty string")
        try:
            annotation_code = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid annotation code: {raw_label!r}") from exc

        match = _SPEAKER_PATTERN.match(raw_turn.strip())
        if not match:
            if not turns:
                raise ValueError(f"cannot parse first utterance: {raw_turn}")
            turns[-1].text = f"{turns[-1].text}\n{_strip_quotes(raw_turn)}"
            turns[-1].annotation_codes.append(annotation_code)
            diagnostics.continuation_lines += 1
            continue

        speaker, text = match.groups()
        if speaker not in USER_SPEAKERS and speaker not in ASSISTANT_SPEAKERS:
            raise ValueError(f"unknown speaker: {speaker}")
        role: Role = "user" if speaker in USER_SPEAKERS else "assistant"
        text = _strip_quotes(text)
        if not text:
            raise ValueError(f"empty utterance for speaker: {speaker}")

        if turns and turns[-1].role == role:
            turns[-1].text = f"{turns[-1].text}\n{text}"
            turns[-1].annotation_codes.append(annotation_code)
            diagnostics.consecutive_turns_merged += 1
        else:
            turns.append(LabeledTurn(role=role, text=text, annotation_codes=[annotation_code]))
    return turns, diagnostics


def prepare_clarification_dataset(raw_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build deterministic weakly supervised clarification-detection data."""

    raw_bytes = raw_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != RAW_SHA256:
        raise ValueError(
            f"raw dataset checksum mismatch: expected {RAW_SHA256}, got {actual_sha256}"
        )
    rows = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(rows, list):
        raise TypeError("raw dataset root must be a JSON list")

    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("raw dataset rows must be JSON objects")
        fingerprint = _row_fingerprint(row)
        unique_rows.setdefault(fingerprint, row)

    candidates: list[dict[str, Any]] = []
    parse_diagnostics = NLUParseDiagnostics()
    redaction_counts: Counter[str] = Counter()
    rows_with_samples: set[str] = set()
    response_code_counts: Counter[int] = Counter()

    for fingerprint, row in sorted(unique_rows.items()):
        source_id = f"car-{fingerprint[:16]}"
        raw_turns = row.get("total_split")
        raw_labels = row.get("split_sentence_type")
        if not isinstance(raw_turns, list) or not isinstance(raw_labels, list):
            raise TypeError(f"{source_id}: total_split and split_sentence_type must be lists")
        if not raw_turns:
            continue

        try:
            turns, diagnostics = parse_labeled_dialogue(raw_turns, raw_labels)
        except ValueError as exc:
            raise ValueError(f"{source_id}: {exc}") from exc
        parse_diagnostics.continuation_lines += diagnostics.continuation_lines
        parse_diagnostics.consecutive_turns_merged += diagnostics.consecutive_turns_merged

        for turn in turns:
            redaction = redact_text(turn.text)
            turn.text = redaction.text
            redaction_counts.update(redaction.counts)

        sample_ordinal = 0
        for index, turn in enumerate(turns[:-1]):
            response = turns[index + 1]
            if turn.role != "user" or response.role != "assistant":
                continue
            label = _weak_label(response.annotation_codes)
            if label is None:
                continue

            sample_ordinal += 1
            rows_with_samples.add(source_id)
            response_code_counts.update(response.annotation_codes)
            context_turns = turns[max(0, index - 3) : index + 1]
            context = "\n".join(
                f"{'顾客' if item.role == 'user' else '客服'}：{item.text}"
                for item in context_turns
            )
            candidates.append(
                {
                    "id": f"{source_id}-clarification-{sample_ordinal:02d}",
                    "text": turn.text,
                    "context": context,
                    "label": label,
                    "label_name": "needs_clarification" if label else "can_answer",
                    "metadata": {
                        "source_id": source_id,
                        "response_annotation_codes": response.annotation_codes,
                        "source_url": SOURCE_URL,
                        "source_revision": SOURCE_REVISION,
                        "license": "MIT",
                    },
                }
            )

    records, deduplication = _deduplicate_candidates(candidates)
    by_split: dict[Split, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    source_ids_by_split: dict[Split, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }
    for record in records:
        source_id = record["metadata"]["source_id"]
        split = assign_split(source_id)
        by_split[split].append(record)
        source_ids_by_split[split].add(source_id)

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

    return {
        "format_version": 1,
        "task": "response-derived binary clarification detection",
        "dataset": {
            "name": "Chinese Ambiguous Reference",
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "license": "MIT",
            "raw_sha256": actual_sha256,
        },
        "weak_label_policy": {
            "positive_name": "needs_clarification",
            "positive_response_codes": sorted(CLARIFICATION_TRIGGER_CODES),
            "negative_name": "can_answer",
            "negative_response_codes": sorted(NON_CLARIFICATION_CODES),
            "explanation": (
                "The target is inferred from the next merged service-response annotations. "
                "It is not a direct human annotation of the user utterance."
            ),
            "known_limitation": (
                "The pinned JSON's code usage is not fully consistent with the repository's "
                "prose label table, so numeric codes and assumptions remain in every record."
            ),
        },
        "statistics": {
            "raw_rows": len(rows),
            "unique_rows": len(unique_rows),
            "duplicate_rows_removed": len(rows) - len(unique_rows),
            "rows_with_samples": len(rows_with_samples),
            "candidate_samples": len(candidates),
            **deduplication,
            "output_samples": len(records),
            "labels": {
                "can_answer": sum(record["label"] == 0 for record in records),
                "needs_clarification": sum(record["label"] == 1 for record in records),
            },
            "samples_by_split": {split: len(by_split[split]) for split in ("train", "dev", "test")},
            "sources_by_split": {
                split: len(source_ids_by_split[split]) for split in ("train", "dev", "test")
            },
            "response_annotation_code_counts": {
                str(code): count for code, count in sorted(response_code_counts.items())
            },
            "redactions": dict(sorted(redaction_counts.items())),
            "parser_recoveries": asdict(parse_diagnostics),
        },
        "outputs": outputs,
    }


def _weak_label(response_codes: list[int]) -> int | None:
    if any(code in CLARIFICATION_TRIGGER_CODES for code in response_codes):
        return 1
    if response_codes and all(code in NON_CLARIFICATION_CODES for code in response_codes):
        return 0
    return None


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["context"]].append(candidate)

    records: list[dict[str, Any]] = []
    duplicate_samples_removed = 0
    conflicting_groups_removed = 0
    conflicting_samples_removed = 0
    for _, group in sorted(grouped.items()):
        labels = {record["label"] for record in group}
        if len(labels) > 1:
            conflicting_groups_removed += 1
            conflicting_samples_removed += len(group)
            continue
        group.sort(key=lambda record: record["id"])
        records.append(group[0])
        duplicate_samples_removed += len(group) - 1

    records.sort(key=lambda record: record["id"])
    return records, {
        "duplicate_samples_removed": duplicate_samples_removed,
        "conflicting_groups_removed": conflicting_groups_removed,
        "conflicting_samples_removed": conflicting_samples_removed,
    }


def _row_fingerprint(row: dict[str, Any]) -> str:
    canonical = {field: row.get(field) for field in _CANONICAL_FIELDS}
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _strip_quotes(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("“", '"')):
        stripped = stripped[1:]
    if stripped.endswith(("”", '"')):
        stripped = stripped[:-1]
    return stripped.strip()
