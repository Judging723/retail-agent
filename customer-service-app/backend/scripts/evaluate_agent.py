import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.agents import CUSTOMER_SERVICE_TOOLS
from app.core.config import get_settings
from app.evaluation import evaluate_agent_cases, load_agent_cases, summarize_agent_results
from app.llm import OpenAICompatibleLLMClient
from app.services.conversation import DEFAULT_SYSTEM_PROMPT

DEFAULT_CASES_PATH = BACKEND_ROOT / "eval" / "agent_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Agent tool selection and arguments")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the fixed dataset without calling the model API",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


async def run_evaluation(args: argparse.Namespace) -> None:
    cases = load_agent_cases(args.cases)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than zero")
        cases = cases[: args.limit]

    if args.validate_only:
        tool_cases = sum(case.expected_tool is not None for case in cases)
        print(f"dataset_valid=true cases={len(cases)} tool_cases={tool_cases}")
        return

    settings = get_settings()
    api_key = settings.llm_api_key.get_secret_value()
    if not api_key:
        raise SystemExit("RETAIL_LLM_API_KEY is required for live evaluation")

    async with AsyncOpenAI(api_key=api_key, base_url=settings.llm_api_base) as sdk_client:
        llm_client = OpenAICompatibleLLMClient(
            client=sdk_client,
            model=settings.llm_model,
        )
        results = await evaluate_agent_cases(cases, llm_client)

    summary = summarize_agent_results(results)
    payload = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "cases_path": _display_path(args.cases),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "system_prompt_sha256": _text_sha256(DEFAULT_SYSTEM_PROMPT),
        "tools_sha256": _text_sha256(
            json.dumps(CUSTOMER_SERVICE_TOOLS, ensure_ascii=False, sort_keys=True)
        ),
        "summary": summary.as_dict(),
        "results": [result.as_dict() for result in results],
    }

    print(f"model={settings.llm_model}")
    print(f"cases={summary.total_cases}")
    print(f"tool_selection_accuracy={summary.tool_selection_accuracy:.3f}")
    print(f"tool_case_end_to_end_accuracy={summary.tool_case_end_to_end_accuracy:.3f}")
    print(f"argument_accuracy_when_tool_correct={summary.argument_accuracy_when_tool_correct:.3f}")
    print(f"no_tool_accuracy={summary.no_tool_accuracy:.3f}")
    for result in results:
        if not result.tool_selection_correct or result.arguments_correct is False:
            print(
                f"FAIL {result.case_id} expected={result.expected_tool} "
                f"predicted={list(result.predicted_tools)} error={result.error}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output={args.output}")


def _display_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(BACKEND_ROOT))
    except ValueError:
        return str(resolved_path)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    asyncio.run(run_evaluation(parse_args()))


if __name__ == "__main__":
    main()
