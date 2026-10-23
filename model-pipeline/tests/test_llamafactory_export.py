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
    exported = json.loads((output_dir / "retail_agent_train.jsonl").read_text(encoding="utf-8"))
    assert exported == {"messages": source_record["messages"]}
    dataset_info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert dataset_info["retail_agent_train"]["formatting"] == "sharegpt"
    assert dataset_info["retail_agent_train"]["tags"]["user_tag"] == "user"


def test_export_adds_repeated_business_boundaries_as_separate_dataset(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_record = {
        "messages": [
            {"role": "user", "content": "我想买件衣服。"},
            {"role": "assistant", "content": "请问您需要什么尺码？"},
        ]
    }
    for split in ("train", "dev", "test"):
        (input_dir / f"{split}.jsonl").write_text(
            json.dumps(source_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    boundary_path = tmp_path / "boundaries.json"
    boundary_record = {
        "id": "refund-without-tool",
        "messages": [
            {"role": "user", "content": "帮我直接退款。"},
            {
                "role": "assistant",
                "content": "请先提供订单号；在退款工具返回成功前，我不能声称退款已完成。",
            },
        ],
    }
    boundary_path.write_text(
        json.dumps([boundary_record], ensure_ascii=False),
        encoding="utf-8",
    )

    counts = export_llamafactory_dataset(
        input_dir,
        output_dir,
        boundary_path=boundary_path,
        boundary_repeat=3,
    )

    assert counts == {"train": 1, "dev": 1, "test": 1, "boundary_train": 3}
    boundary_lines = (
        (output_dir / "retail_agent_boundary_train.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(boundary_lines) == 3
    assert all(
        json.loads(line) == {"messages": boundary_record["messages"]} for line in boundary_lines
    )

    dataset_info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert dataset_info["retail_agent_boundary_train"]["file_name"] == (
        "retail_agent_boundary_train.jsonl"
    )
