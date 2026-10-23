import hashlib
import json
from pathlib import Path

from app.agents import CUSTOMER_SERVICE_TOOLS
from app.evaluation import (
    AgentEvalCase,
    AgentEvalResult,
    evaluate_agent_turn,
    load_agent_cases,
    summarize_agent_results,
)
from app.llm import AssistantTurn, FunctionToolCall
from app.services.conversation import DEFAULT_SYSTEM_PROMPT

CASES_PATH = Path(__file__).resolve().parents[1] / "eval" / "agent_cases.json"
BASELINE_PATH = CASES_PATH.parent / "results" / "deepseek-baseline.json"


def test_fixed_agent_evaluation_dataset_is_valid() -> None:
    cases = load_agent_cases(CASES_PATH)

    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert sum(case.expected_tool is not None for case in cases) == 12
    assert sum(case.expected_tool is None for case in cases) == 8


def test_agent_evaluation_normalizes_refund_amounts() -> None:
    case = AgentEvalCase.model_validate(
        {
            "case_id": "refund_normalization",
            "category": "refund_request",
            "messages": [{"role": "user", "content": "提交退款"}],
            "expected_tool": "request_refund",
            "expected_arguments": {
                "order_no": "ORDER-1",
                "amount": "50.00",
                "reason": "商品破损",
            },
        }
    )
    turn = AssistantTurn(
        content=None,
        tool_calls=(
            FunctionToolCall(
                id="call-1",
                name="request_refund",
                arguments=json.dumps(
                    {
                        "order_no": "ORDER-1",
                        "amount": 50,
                        "reason": "商品破损",
                    }
                ),
            ),
        ),
    )

    result = evaluate_agent_turn(case, turn)

    assert result.tool_selection_correct is True
    assert result.arguments_correct is True


def test_agent_evaluation_separates_selection_arguments_and_no_tool() -> None:
    order_case = AgentEvalCase.model_validate(
        {
            "case_id": "order_case",
            "category": "order_query",
            "messages": [{"role": "user", "content": "查询订单"}],
            "expected_tool": "get_order_status",
            "expected_arguments": {"order_no": "ORDER-1"},
        }
    )
    no_tool_case = AgentEvalCase.model_validate(
        {
            "case_id": "no_tool_case",
            "category": "no_tool",
            "messages": [{"role": "user", "content": "你好"}],
            "expected_tool": None,
            "expected_arguments": None,
        }
    )
    invalid_arguments_turn = AssistantTurn(
        content=None,
        tool_calls=(
            FunctionToolCall(
                id="call-2",
                name="get_order_status",
                arguments='{"order_no":"包含 空格"}',
            ),
        ),
    )
    no_tool_turn = AssistantTurn(content="您好", tool_calls=())

    results = [
        evaluate_agent_turn(order_case, invalid_arguments_turn),
        evaluate_agent_turn(no_tool_case, no_tool_turn),
    ]
    summary = summarize_agent_results(results)

    assert summary.tool_selection_accuracy == 1.0
    assert summary.tool_case_end_to_end_accuracy == 0.0
    assert summary.argument_accuracy_when_tool_correct == 0.0
    assert summary.no_tool_accuracy == 1.0
    assert results[0].error is not None


def test_versioned_baseline_matches_cases_prompt_tools_and_metrics() -> None:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    results = [
        AgentEvalResult(
            case_id=item["case_id"],
            category=item["category"],
            expected_tool=item["expected_tool"],
            predicted_tools=tuple(item["predicted_tools"]),
            tool_selection_correct=item["tool_selection_correct"],
            arguments_correct=item["arguments_correct"],
            predicted_arguments=item["predicted_arguments"],
            error=item["error"],
        )
        for item in payload["results"]
    ]

    assert summarize_agent_results(results).as_dict() == payload["summary"]
    assert payload["cases_sha256"] == hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()
    assert (
        payload["system_prompt_sha256"]
        == hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    )
    serialized_tools = json.dumps(CUSTOMER_SERVICE_TOOLS, ensure_ascii=False, sort_keys=True)
    assert payload["tools_sha256"] == hashlib.sha256(serialized_tools.encode("utf-8")).hexdigest()
