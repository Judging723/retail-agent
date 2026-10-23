import argparse
import json
from pathlib import Path

from model_pipeline.entity_baseline import evaluate_entity_cases, load_entity_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data" / "entity_cases.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "entity_baseline.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the minimal order-entity baseline")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = evaluate_entity_cases(load_entity_cases(args.cases))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"cases={report['cases']}")
    print(f"exact_match_accuracy={report['exact_match_accuracy']:.3f}")
    print(f"micro_f1={report['micro']['f1']:.3f}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
