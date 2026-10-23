import argparse
import json
from pathlib import Path

from model_pipeline.sentiment_baseline import train_and_evaluate_sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sentiment"
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "sentiment" / "sentiment_tfidf.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "sentiment_baseline.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate sentiment baselines")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = train_and_evaluate_sentiment(args.data_dir, args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for split, results in report["results"].items():
        print(f"{split}:")
        for model_name, metrics in results.items():
            print(
                f"  {model_name}: accuracy={metrics['accuracy']:.3f} "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )
    print(f"artifact={args.artifact}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
