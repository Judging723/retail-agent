import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents import CUSTOMER_SERVICE_TOOLS
from app.agents.order_tools import GetOrderStatusArguments, RequestRefundArguments
from app.llm import AssistantTurn, ChatMessage, LLMClient
from app.prompts import DEFAULT_SYSTEM_PROMPT

ExpectedTool = Literal["get_order_status", "request_refund"]


class EvalMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AgentEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    category: str = Field(min_length=1, max_length=64)
    messages: list[EvalMessage] = Field(min_length=1)
    expected_tool: ExpectedTool | None
    expected_arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_expected_action(self) -> "AgentEvalCase":
        if self.messages[-1].role != "user":
            raise ValueError("the final evaluation message must come from the user")
        if self.expected_tool is None:
            if self.expected_arguments is not None:
                raise ValueError("no-tool cases cannot define expected_arguments")
            return self
        if self.expected_arguments is None:
            raise ValueError("tool cases must define expected_arguments")
        _parse_arguments(self.expected_tool, self.expected_arguments)
        return self


@dataclass(frozen=True, slots=True)
class AgentEvalResult:
    case_id: str
    category: str
    expected_tool: str | None
    predicted_tools: tuple[str, ...]
    tool_selection_correct: bool
    arguments_correct: bool | None
    predicted_arguments: dict[str, Any] | None
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["predicted_tools"] = list(self.predicted_tools)
        return result


@dataclass(frozen=True, slots=True)
class AgentEvalSummary:
    total_cases: int
    tool_selection_accuracy: float
    tool_cases: int
    tool_case_end_to_end_accuracy: float
    correct_tool_cases: int
    argument_accuracy_when_tool_correct: float
    no_tool_cases: int
    no_tool_accuracy: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_agent_cases(path: Path) -> list[AgentEvalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    cases = [AgentEvalCase.model_validate(item) for item in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case_id values must be unique")
    return cases


def evaluate_agent_turn(case: AgentEvalCase, turn: AssistantTurn) -> AgentEvalResult:
    predicted_tools = tuple(tool_call.name for tool_call in turn.tool_calls)
    expected_tools = () if case.expected_tool is None else (case.expected_tool,)
    selection_correct = predicted_tools == expected_tools

    arguments_correct: bool | None = None
    predicted_arguments: dict[str, Any] | None = None
    error: str | None = None
    if case.expected_tool is not None:
        arguments_correct = False
        if len(turn.tool_calls) == 1 and turn.tool_calls[0].name == case.expected_tool:
            try:
                predicted_model = _parse_arguments(
                    case.expected_tool,
                    turn.tool_calls[0].arguments,
                )
                expected_model = _parse_arguments(
                    case.expected_tool,
                    case.expected_arguments,
                )
                predicted_arguments = predicted_model.model_dump(mode="json")
                arguments_correct = predicted_model == expected_model
            except (ValueError, TypeError) as exc:
                error = str(exc)

    return AgentEvalResult(
        case_id=case.case_id,
        category=case.category,
        expected_tool=case.expected_tool,
        predicted_tools=predicted_tools,
        tool_selection_correct=selection_correct,
        arguments_correct=arguments_correct,
        predicted_arguments=predicted_arguments,
        error=error,
    )


async def evaluate_agent_cases(
    cases: Sequence[AgentEvalCase],
    llm_client: LLMClient,
) -> list[AgentEvalResult]:
    results = []
    for case in cases:
        messages: list[ChatMessage] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            *[message.model_dump() for message in case.messages],
        ]
        turn = await llm_client.complete_with_tools(messages, CUSTOMER_SERVICE_TOOLS)
        results.append(evaluate_agent_turn(case, turn))
    return results


def summarize_agent_results(results: Sequence[AgentEvalResult]) -> AgentEvalSummary:
    total_cases = len(results)
    tool_results = [result for result in results if result.expected_tool is not None]
    correct_tool_results = [result for result in tool_results if result.tool_selection_correct]
    no_tool_results = [result for result in results if result.expected_tool is None]
    return AgentEvalSummary(
        total_cases=total_cases,
        tool_selection_accuracy=_ratio(
            sum(result.tool_selection_correct for result in results),
            total_cases,
        ),
        tool_cases=len(tool_results),
        tool_case_end_to_end_accuracy=_ratio(
            sum(result.arguments_correct is True for result in tool_results),
            len(tool_results),
        ),
        correct_tool_cases=len(correct_tool_results),
        argument_accuracy_when_tool_correct=_ratio(
            sum(result.arguments_correct is True for result in correct_tool_results),
            len(correct_tool_results),
        ),
        no_tool_cases=len(no_tool_results),
        no_tool_accuracy=_ratio(
            sum(result.tool_selection_correct for result in no_tool_results),
            len(no_tool_results),
        ),
    )


def _parse_arguments(
    tool_name: str,
    arguments: str | dict[str, Any] | None,
) -> BaseModel:
    if arguments is None:
        raise ValueError("tool arguments are missing")
    model_type = {
        "get_order_status": GetOrderStatusArguments,
        "request_refund": RequestRefundArguments,
    }.get(tool_name)
    if model_type is None:
        raise ValueError(f"unsupported tool: {tool_name}")
    if isinstance(arguments, str):
        return model_type.model_validate_json(arguments)
    return model_type.model_validate(arguments)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
