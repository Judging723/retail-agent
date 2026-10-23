import json
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")


def export_llamafactory_dataset(
    input_dir: Path,
    output_dir: Path,
    boundary_path: Path | None = None,
    boundary_repeat: int = 1,
) -> dict[str, int]:
    """Export prepared chat records to LLaMA-Factory's OpenAI message format."""

    if boundary_repeat < 1:
        raise ValueError("boundary_repeat must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    dataset_info: dict[str, Any] = {}

    for split in SPLITS:
        source_path = input_dir / f"{split}.jsonl"
        output_path = output_dir / f"retail_agent_{split}.jsonl"
        exported: list[dict[str, Any]] = []

        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            record = json.loads(line)
            messages = record.get("messages")
            _validate_messages(messages, source_path, line_number)
            exported.append({"messages": messages})

        output_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in exported
            ),
            encoding="utf-8",
        )
        counts[split] = len(exported)
        dataset_info[f"retail_agent_{split}"] = _dataset_info(output_path.name)

    if boundary_path is not None:
        boundary_records = json.loads(boundary_path.read_text(encoding="utf-8"))
        if not isinstance(boundary_records, list):
            raise TypeError("business boundary data must be a JSON list")

        exported_boundaries: list[dict[str, Any]] = []
        for index, record in enumerate(boundary_records, start=1):
            messages = record.get("messages") if isinstance(record, dict) else None
            _validate_messages(messages, boundary_path, index)
            exported_boundaries.extend({"messages": messages} for _ in range(boundary_repeat))

        boundary_output = output_dir / "retail_agent_boundary_train.jsonl"
        boundary_output.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in exported_boundaries
            ),
            encoding="utf-8",
        )
        counts["boundary_train"] = len(exported_boundaries)
        dataset_info["retail_agent_boundary_train"] = _dataset_info(boundary_output.name)

    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def _dataset_info(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _validate_messages(
    messages: Any,
    source_path: Path,
    line_number: int,
) -> None:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{source_path}:{line_number}: messages must contain a dialogue")
    if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        raise ValueError(f"{source_path}:{line_number}: invalid first or last role")
    for index, message in enumerate(messages):
        if message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"{source_path}:{line_number}: unsupported message role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{source_path}:{line_number}: empty message content")
        if index and messages[index - 1]["role"] == message["role"]:
            raise ValueError(f"{source_path}:{line_number}: message roles must alternate")
