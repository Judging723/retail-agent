import json
from pathlib import Path

from model_pipeline.preparation import assign_split, parse_dialogue, prepare_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DATA = REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"


def test_parse_dialogue_recovers_known_public_dataset_noise() -> None:
    content = """场景背景：测试
真实对话：
客服：“请问需要什么？”
顾客“想买红色的
而且要大一点”
客服：好的，我帮您找红色大号。
"""

    messages, diagnostics = parse_dialogue(content)

    assert [(message.role, message.content) for message in messages] == [
        ("assistant", "请问需要什么？"),
        ("user", "想买红色的\n而且要大一点"),
        ("assistant", "好的，我帮您找红色大号。"),
    ]
    assert diagnostics.missing_separator_lines == 1
    assert diagnostics.continuation_lines == 1


def test_assign_split_is_stable_for_the_same_source() -> None:
    assert assign_split("car-example") == assign_split("car-example")
    assert assign_split("car-example") in {"train", "dev", "test"}


def test_public_dataset_preparation_is_reproducible_and_leak_free(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_dataset(PUBLIC_DATA, first_dir)
    second = prepare_dataset(PUBLIC_DATA, second_dir)

    assert first == second
    assert first["statistics"]["source_documents"] == 1073
    assert first["statistics"]["output_samples"] > 4000
    assert first["statistics"]["duplicates_removed"] > 0
    assert all(
        first["statistics"]["samples_by_split"][name] > 0 for name in ("train", "dev", "test")
    )

    source_splits: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for line in (first_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            messages = record["messages"]
            source_id = record["metadata"]["source_id"]
            assert messages[0]["role"] == "user"
            assert messages[-1]["role"] == "assistant"
            assert source_splits.setdefault(source_id, split) == split
