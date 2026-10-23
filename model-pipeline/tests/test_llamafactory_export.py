import json
from pathlib import Path

from model_pipeline.llamafactory_export import export_llamafactory_dataset


def test_export_llamafactory_dataset_keeps_only_messages(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_record = {
        "id": "sample-01",
        "messages": [
            {"role": "user", "content": "我想买件衣服。"},
            {"role": "assistant", "content": "请问您需要什么尺码？"},
        ],
        "metadata": {"source_id": "source-01"},
    }
    for split in ("train", "dev", "test"):
        (input_dir / f"{split}.jsonl").write_text(
            json.dumps(source_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    counts = export_llamafactory_dataset(input_dir, output_dir)

    assert counts == {"train": 1, "dev": 1, "test": 1}
    exported = json.loads(
        (output_dir / "retail_agent_train.jsonl").read_text(encoding="utf-8")
    )
    assert exported == {"messages": source_record["messages"]}
    dataset_info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert dataset_info["retail_agent_train"]["formatting"] == "sharegpt"
    assert dataset_info["retail_agent_train"]["tags"]["user_tag"] == "user"
