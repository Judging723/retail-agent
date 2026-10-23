import json
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")


def export_llamafactory_dataset(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Export prepared chat records to LLaMA-Factory's OpenAI message format."""

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
        dataset_info[f"retail_agent_{split}"] = {
            "file_name": output_path.name,
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

    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


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
