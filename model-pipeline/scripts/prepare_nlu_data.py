import argparse
import json
from pathlib import Path

from model_pipeline.nlu_dataset import prepare_clarification_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_RAW_PATH = (
    REPOSITORY_ROOT
    / "knowledge-pipeline"
    / "data"
    / "raw"
    / "chinese_ambiguous_reference"
    / "All Conversation.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nlu"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "nlu_dataset_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare weakly supervised clarification-detection data"
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = prepare_clarification_dataset(args.raw_path, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    statistics = manifest["statistics"]
    print(f"candidate_samples={statistics['candidate_samples']}")
    print(f"output_samples={statistics['output_samples']}")
    print(f"labels={statistics['labels']}")
    print(f"samples_by_split={statistics['samples_by_split']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
