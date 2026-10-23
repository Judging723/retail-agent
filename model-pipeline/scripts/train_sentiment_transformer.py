import argparse
import json
from pathlib import Path

from model_pipeline.transformer_sentiment import train_transformer_sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "sentiment" / "rbt3"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "sentiment_transformer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the Chinese sentiment Transformer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    report = train_transformer_sentiment(
        args.data_dir,
        args.artifact_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for split in ("dev", "test"):
        metrics = report["results"][split]["metrics"]
        print(f"{split}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
    print(f"artifact={args.artifact_dir}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
