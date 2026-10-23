import argparse
import json
from pathlib import Path

from model_pipeline.preparation import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE = REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "sft_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic, redacted SFT data from the public shopping dialogues"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = prepare_dataset(args.source, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    stats = manifest["statistics"]
    print(f"source_documents={stats['source_documents']}")
    print(f"candidate_samples={stats['candidate_samples']}")
    print(f"duplicates_removed={stats['duplicates_removed']}")
    print(f"output_samples={stats['output_samples']}")
    print(f"samples_by_split={stats['samples_by_split']}")
    print(f"redactions={stats['redactions']}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
