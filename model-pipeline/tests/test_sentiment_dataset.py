import json
from pathlib import Path

import pytest

from model_pipeline.sentiment_dataset import prepare_sentiment_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kapibala_sales_dialogues"
META_PATH = RAW_DIR / "meta.jsonl"
CONVERSATIONS_PATH = RAW_DIR / "conversations_L3.jsonl"
requires_public_raw_data = pytest.mark.skipif(
    not META_PATH.exists() or not CONVERSATIONS_PATH.exists(),
    reason="download the pinned Kapibala source files first",
)


@requires_public_raw_data
def test_sentiment_dataset_is_reproducible_grouped_and_deduplicated(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_sentiment_dataset(META_PATH, CONVERSATIONS_PATH, first_dir)
    second = prepare_sentiment_dataset(META_PATH, CONVERSATIONS_PATH, second_dir)

    assert first == second
    assert first["statistics"]["output_messages"] == 3264
    assert first["statistics"]["duplicates_removed"] == 291
    assert first["statistics"]["labels"] == {
        "negative": 1393,
        "neutral": 326,
        "positive": 1545,
    }

    group_splits: dict[str, str] = {}
    seen_texts: set[str] = set()
    for split in ("train", "dev", "test"):
        for line in (first_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            setting_id = record["metadata"]["setting_id"]
            assert group_splits.setdefault(setting_id, split) == split
            assert record["text"] not in seen_texts
            seen_texts.add(record["text"])
