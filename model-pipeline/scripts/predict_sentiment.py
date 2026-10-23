import argparse
import json
from pathlib import Path

from model_pipeline.sentiment_inference import SentimentPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "sentiment" / "rbt3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment with the local trained model")
    parser.add_argument("text", help="Chinese customer message")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    predictor = SentimentPredictor(args.model_dir)
    print(json.dumps(predictor.predict(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
