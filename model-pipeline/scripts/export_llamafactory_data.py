import argparse
from pathlib import Path

from model_pipeline.llamafactory_export import export_llamafactory_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "llamafactory"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SFT data for LLaMA-Factory")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    counts = export_llamafactory_dataset(args.input_dir, args.output_dir)
    print(f"samples_by_split={counts}")
    print(f"dataset_dir={args.output_dir}")


if __name__ == "__main__":
    main()
