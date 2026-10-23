import json
from pathlib import Path

import pytest

from model_pipeline.nlu_dataset import parse_labeled_dialogue, prepare_clarification_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = (
    REPOSITORY_ROOT
    / "knowledge-pipeline"
    / "data"
    / "raw"
    / "chinese_ambiguous_reference"
    / "All Conversation.json"
)
requires_public_raw_data = pytest.mark.skipif(
    not RAW_DATA.exists(),
    reason="run knowledge-pipeline/scripts/prepare_public_knowledge.py first",
)


def test_parse_labeled_dialogue_merges_role_messages_and_wrapped_lines() -> None:
    turns, diagnostics = parse_labeled_dialogue(
        [
            "顾客：“我想买手机。”",
            "顾客：“请问有什么推荐？”",
            "客服：“可以先看这款",
            "续行内容。”",
        ],
        [0, 0, 3, 3],
    )

    assert [(turn.role, turn.text, turn.annotation_codes) for turn in turns] == [
        ("user", "我想买手机。\n请问有什么推荐？", [0, 0]),
        ("assistant", "可以先看这款\n续行内容。", [3, 3]),
    ]
    assert diagnostics.consecutive_turns_merged == 1
    assert diagnostics.continuation_lines == 1


@requires_public_raw_data
def test_public_nlu_dataset_is_balanced_reproducible_and_leak_free(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_clarification_dataset(RAW_DATA, first_dir)
    second = prepare_clarification_dataset(RAW_DATA, second_dir)

    assert first == second
    assert first["statistics"]["output_samples"] == 3961
    assert first["statistics"]["labels"] == {
        "can_answer": 1884,
        "needs_clarification": 2077,
    }

    source_splits: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for line in (first_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            source_id = record["metadata"]["source_id"]
            assert record["label"] in {0, 1}
            assert source_splits.setdefault(source_id, split) == split
