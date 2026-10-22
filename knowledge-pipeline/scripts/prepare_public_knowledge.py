import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_PATH = (
    PROJECT_ROOT / "data" / "raw" / "chinese_ambiguous_reference" / "All Conversation.json"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"
)

DATASET_NAME = "Chinese Ambiguous Reference"
SOURCE_REPOSITORY = "https://github.com/ygan/Chinese-Ambiguous-Reference"
SOURCE_REVISION = "3fe3a42233571914ee4a5ccaec3f3af97966d9c9"
RAW_URL = (
    "https://raw.githubusercontent.com/ygan/Chinese-Ambiguous-Reference/"
    f"{SOURCE_REVISION}/All%20Conversation.json"
)
RAW_SHA256 = "36631734421c1b892b30e9f7298ab1f29bfc1de5ccc59fe14bbc91a1256377fe"

LABEL_NAMES = {
    0: "顾客表达",
    1: "澄清提问",
    2: "列举选项",
    3: "歧义表达",
    4: "列举并澄清",
    5: "信息收集",
    6: "忽略澄清需求",
    7: "基于歧义收集信息",
    8: "其他",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert the public Chinese Ambiguous Reference dataset"
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    raw_bytes = _read_or_download(args.raw_path)
    _verify_sha256(raw_bytes)
    raw_rows = json.loads(raw_bytes.decode("utf-8"))
    documents = convert_rows(raw_rows)

    payload = {
        "dataset_name": DATASET_NAME,
        "description": (
            "真实购物场景中的中文歧义识别与澄清对话案例。"
            "该数据用于学习澄清策略，不代表任何平台的退款、物流或售后规则。"
        ),
        "source_url": SOURCE_REPOSITORY,
        "license": "MIT",
        "source_revision": SOURCE_REVISION,
        "raw_sha256": RAW_SHA256,
        "raw_record_count": len(raw_rows),
        "document_count": len(documents),
        "documents": documents,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"raw_records={len(raw_rows)}")
    print(f"documents={len(documents)}")
    print(f"duplicate_or_empty_records_removed={len(raw_rows) - len(documents)}")
    print(f"output={args.output_path}")


def convert_rows(raw_rows: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        raise ValueError("dataset root must be a JSON list")

    documents: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()

    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be a JSON object")

        canonical_row = {
            field: row.get(field)
            for field in (
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
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                canonical_row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)

        merchandise = _text(row.get("merchandise"), fallback="未标注商品")
        shop = _text(row.get("shop"), fallback="未标注店铺")
        location = _text(row.get("location"), fallback="未标注地区")
        source_date = _text(row.get("date"), fallback="未标注")
        background = _text(row.get("shopBackground"), fallback="未提供场景背景")
        dialogue = _dialogue_text(row)
        if not dialogue:
            continue
        labels = _label_names(row.get("split_sentence_type"))

        documents.append(
            {
                "source_id": f"car-{fingerprint[:16]}",
                "title": f"购物澄清案例：{merchandise}",
                "category": "购物歧义与澄清",
                "content": (
                    f"商品或服务：{merchandise}\n"
                    f"店铺类型：{shop}\n"
                    f"发生地区：{location}\n"
                    f"场景背景：{background}\n"
                    f"对话标注：{'、'.join(labels)}\n"
                    f"真实对话：\n{dialogue}"
                ),
                "keywords": _unique_nonempty([merchandise, shop, location, "购物咨询", "澄清"]),
                "revision": SOURCE_REVISION[:12],
                "source_date": source_date,
                "source_url": SOURCE_REPOSITORY,
                "license": "MIT",
            }
        )

    return documents


def _read_or_download(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading={RAW_URL}")
    with urlopen(RAW_URL, timeout=60) as response:
        raw_bytes = response.read()
    path.write_bytes(raw_bytes)
    return raw_bytes


def _verify_sha256(raw_bytes: bytes) -> None:
    actual = hashlib.sha256(raw_bytes).hexdigest()
    if actual != RAW_SHA256:
        raise ValueError(f"dataset checksum mismatch: expected {RAW_SHA256}, got {actual}")


def _dialogue_text(row: dict[str, Any]) -> str:
    split = row.get("total_split")
    if isinstance(split, list):
        lines = [_text(line) for line in split]
        lines = [line for line in lines if line]
        if lines:
            return "\n".join(lines)

    total = _text(row.get("total"))
    return total


def _label_names(raw_labels: Any) -> list[str]:
    if not isinstance(raw_labels, list):
        return ["未标注"]

    normalized: list[int] = []
    for label in raw_labels:
        try:
            value = int(label)
        except (TypeError, ValueError):
            continue
        if value not in normalized:
            normalized.append(value)

    return [LABEL_NAMES.get(label, f"未知标签 {label}") for label in normalized] or ["未标注"]


def _text(value: Any, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    return value.strip() or fallback


def _unique_nonempty(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))


if __name__ == "__main__":
    main()
