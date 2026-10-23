import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from model_pipeline.redaction import detect_pii, redact_text

Role = Literal["user", "assistant"]
Split = Literal["train", "dev", "test"]

USER_SPEAKERS = frozenset({"顾客", "顾客人", "客户", "买家", "消费者", "用户", "客人", "旅客"})
ASSISTANT_SPEAKERS = frozenset(
    {
        "客服",
        "餐饮服务员",
        "售货员",
        "售卖员",
        "食堂服务员",
        "店员",
        "服务员",
        "摊主",
        "销售员",
        "销售",
        "销售顾问",
        "花店服务员",
        "美甲师",
        "旅行社员工",
        "健身房员工",
        "理发师",
    }
)
KNOWN_SPEAKERS = USER_SPEAKERS | ASSISTANT_SPEAKERS
_MISSING_SEPARATOR = re.compile(
    rf"^({'|'.join(sorted(KNOWN_SPEAKERS, key=len, reverse=True))})[“\"](.*)$"
)
_WITH_SEPARATOR = re.compile(r"^([^：:]{1,30})[：:]\s*(.*)$")


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class ParseDiagnostics:
    missing_separator_lines: int = 0
    continuation_lines: int = 0
    unbalanced_quote_lines: int = 0
    leading_assistant_documents_skipped: int = 0

    def add(self, other: "ParseDiagnostics") -> None:
        for field_name in asdict(self):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


def assign_split(source_id: str) -> Split:
    """Assign every source conversation to one stable 80/10/10 split."""

    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def parse_dialogue(content: str) -> tuple[list[ChatMessage], ParseDiagnostics]:
    """Parse the public dataset's noisy dialogue text into canonical chat messages."""

    marker = "真实对话："
    if marker not in content:
        raise ValueError("missing dialogue marker")

    diagnostics = ParseDiagnostics()
    messages: list[ChatMessage] = []

    for raw_line in content.split(marker, maxsplit=1)[1].splitlines():
        line = raw_line.strip()
        if not line:
            continue

        separator_match = _WITH_SEPARATOR.match(line)
        missing_separator_match = _MISSING_SEPARATOR.match(line)
        if separator_match:
            speaker, text = separator_match.groups()
        elif missing_separator_match:
            speaker, text = missing_separator_match.groups()
            diagnostics.missing_separator_lines += 1
        elif messages:
            previous = messages[-1]
            continuation = _strip_edge_quotes(line)
            messages[-1] = ChatMessage(
                role=previous.role,
                content=f"{previous.content}\n{continuation}",
            )
            diagnostics.continuation_lines += 1
            continue
        else:
            raise ValueError(f"cannot parse dialogue line: {line}")

        speaker = speaker.strip()
        if speaker not in KNOWN_SPEAKERS:
            raise ValueError(f"unknown speaker: {speaker}")

        text = text.strip()
        if text.startswith(("“", '"')) != text.endswith(("”", '"')):
            diagnostics.unbalanced_quote_lines += 1
        text = _strip_edge_quotes(text)
        if not text:
            raise ValueError(f"empty utterance for speaker: {speaker}")

        role: Role = "user" if speaker in USER_SPEAKERS else "assistant"
        if messages and messages[-1].role == role:
            previous = messages[-1]
            messages[-1] = ChatMessage(role=role, content=f"{previous.content}\n{text}")
        else:
            messages.append(ChatMessage(role=role, content=text))

    if not messages:
        raise ValueError("dialogue contains no messages")
    return messages, diagnostics


def prepare_dataset(source_path: Path, output_dir: Path) -> dict[str, Any]:
    """Create deterministic SFT JSONL files and return their reproducibility manifest."""

    source_bytes = source_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise TypeError("source documents must be a JSON list")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[Split, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    source_ids: dict[Split, set[str]] = {"train": set(), "dev": set(), "test": set()}
    output_source_ids: dict[Split, set[str]] = {"train": set(), "dev": set(), "test": set()}
    diagnostics = ParseDiagnostics()
    redaction_counts: Counter[str] = Counter()
    seen_fingerprints: set[str] = set()
    candidate_samples = 0
    duplicates_removed = 0

    for document in sorted(documents, key=lambda item: item["source_id"]):
        source_id = _required_text(document, "source_id")
        split = assign_split(source_id)
        source_ids[split].add(source_id)

        try:
            messages, document_diagnostics = parse_dialogue(_required_text(document, "content"))
        except ValueError as exc:
            raise ValueError(f"{source_id}: {exc}") from exc
        diagnostics.add(document_diagnostics)

        if messages[0].role == "assistant":
            diagnostics.leading_assistant_documents_skipped += 1
            continue

        redacted_messages: list[ChatMessage] = []
        for message in messages:
            result = redact_text(message.content)
            redaction_counts.update(result.counts)
            redacted_messages.append(ChatMessage(role=message.role, content=result.text))

        assistant_ordinal = 0
        for index, message in enumerate(redacted_messages):
            if message.role != "assistant":
                continue
            assistant_ordinal += 1
            candidate_samples += 1
            sample_messages = redacted_messages[: index + 1]
            _validate_messages(sample_messages)
            fingerprint = _messages_fingerprint(sample_messages)
            if fingerprint in seen_fingerprints:
                duplicates_removed += 1
                continue
            seen_fingerprints.add(fingerprint)

            sample = {
                "id": f"{source_id}-response-{assistant_ordinal:02d}",
                "messages": [asdict(item) for item in sample_messages],
                "metadata": {
                    "source_id": source_id,
                    "source_url": payload.get("source_url", document.get("source_url")),
                    "source_revision": payload.get("source_revision", document.get("revision")),
                    "license": payload.get("license", document.get("license")),
                },
            }
            records[split].append(sample)
            output_source_ids[split].add(source_id)

    output_metadata: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        path = output_dir / f"{split}.jsonl"
        serialized = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records[split]
        ).encode("utf-8")
        path.write_bytes(serialized)
        output_metadata[split] = {
            "file": path.name,
            "records": len(records[split]),
            "sha256": hashlib.sha256(serialized).hexdigest(),
        }

    return {
        "format_version": 1,
        "dataset": {
            "name": payload.get("dataset_name"),
            "source_url": payload.get("source_url"),
            "source_revision": payload.get("source_revision"),
            "license": payload.get("license"),
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "transformation": {
            "sample_unit": "one supervised sample per assistant response",
            "split_strategy": "sha256(source_id) bucket: train 0-79, dev 80-89, test 90-99",
            "deduplication": "sha256 of canonical redacted messages",
            "pii_policy": "high-confidence regex pseudonymization before output",
            "leading_assistant_policy": (
                "skip the complete source dialogue instead of removing its opening assistant turn"
            ),
        },
        "statistics": {
            "source_documents": len(documents),
            "candidate_samples": candidate_samples,
            "duplicates_removed": duplicates_removed,
            "output_samples": sum(len(items) for items in records.values()),
            "source_documents_by_split": {
                split: len(source_ids[split]) for split in ("train", "dev", "test")
            },
            "output_sources_by_split": {
                split: len(output_source_ids[split]) for split in ("train", "dev", "test")
            },
            "samples_by_split": {split: len(records[split]) for split in ("train", "dev", "test")},
            "redactions": dict(sorted(redaction_counts.items())),
            "parser_recoveries": asdict(diagnostics),
        },
        "outputs": output_metadata,
    }


def _strip_edge_quotes(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("“", '"')):
        stripped = stripped[1:]
    if stripped.endswith(("”", '"')):
        stripped = stripped[:-1]
    return stripped.strip()


def _validate_messages(messages: list[ChatMessage]) -> None:
    if len(messages) < 2 or messages[0].role != "user" or messages[-1].role != "assistant":
        raise ValueError("sample must start with user and end with assistant")
    for index, message in enumerate(messages):
        if not message.content.strip():
            raise ValueError("sample contains an empty message")
        if detect_pii(message.content):
            raise ValueError("sample still contains detectable PII after redaction")
        if index and messages[index - 1].role == message.role:
            raise ValueError("sample roles must alternate")


def _messages_fingerprint(messages: list[ChatMessage]) -> str:
    canonical = json.dumps(
        [asdict(message) for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _required_text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"document field {field!r} must be a non-empty string")
    return value.strip()
