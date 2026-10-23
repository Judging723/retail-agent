import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from model_pipeline.sentiment_dataset import (
    CONVERSATIONS_L3_SHA256,
    CONVERSATIONS_L3_URL,
    META_SHA256,
    META_URL,
    prepare_sentiment_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kapibala_sales_dialogues"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "sentiment_dataset_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Chinese sentiment pseudo-label data")
    parser.add_argument("--meta", type=Path, default=DEFAULT_RAW_DIR / "meta.jsonl")
    parser.add_argument(
        "--conversations",
        type=Path,
        default=DEFAULT_RAW_DIR / "conversations_L3.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    _download_if_missing(args.meta, META_URL, META_SHA256)
    _download_if_missing(args.conversations, CONVERSATIONS_L3_URL, CONVERSATIONS_L3_SHA256)
    manifest = prepare_sentiment_dataset(args.meta, args.conversations, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    statistics = manifest["statistics"]
    print(f"candidate_messages={statistics['candidate_messages']}")
    print(f"output_messages={statistics['output_messages']}")
    print(f"labels={statistics['labels']}")
    print(f"samples_by_split={statistics['samples_by_split']}")
    print(f"report={args.report}")


def _download_if_missing(path: Path, url: str, expected_sha256: str) -> None:
    if path.exists():
        return
    print(f"downloading={url}")
    with urlopen(url, timeout=120) as response:
        data = response.read()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"download checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


if __name__ == "__main__":
    main()
