import json
import logging
from collections.abc import AsyncIterator
from time import perf_counter

from app.agents.order_tools import CUSTOMER_SERVICE_TOOLS, OrderToolExecutor
from app.llm import ChatMessage, FunctionToolCall, LLMClient, LLMUnavailableError

MAX_TOOL_ROUNDS = 3
logger = logging.getLogger("app.agent")


class CustomerSupportAgent:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        order_tool_executor: OrderToolExecutor,
    ) -> None:
        self._llm_client = llm_client
        self._order_tool_executor = order_tool_executor

    async def respond(
        self,
        messages: list[ChatMessage],
        *,
        conversation_id: int,
        customer_id: int,
        idempotency_key: str,
    ) -> str:
        working_messages = list(messages)

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            turn = await self._llm_client.complete_with_tools(
                working_messages,
                CUSTOMER_SERVICE_TOOLS,
            )

            if not turn.tool_calls:
                if turn.content is None:
                    raise LLMUnavailableError("Model service returned an empty response")
                return turn.content

            if tool_round == MAX_TOOL_ROUNDS:
                raise LLMUnavailableError("Model exceeded the tool call limit")

            working_messages.append(
                {
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in turn.tool_calls
                    ],
                }
            )

            for tool_call in turn.tool_calls:
                tool_result = await self._execute_tool_call(
                    tool_call,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    idempotency_key=idempotency_key,
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": tool_result,
                    }
                )

        raise LLMUnavailableError("Model exceeded the tool call limit")

    async def stream_respond(
        self,
        messages: list[ChatMessage],
        *,
        conversation_id: int,
        customer_id: int,
        idempotency_key: str,
    ) -> AsyncIterator[str]:
        working_messages = list(messages)

        for tool_round in range(MAX_TOOL_ROUNDS + 1):
            completed_turn = None
            async for event in self._llm_client.stream_with_tools(
                working_messages,
                CUSTOMER_SERVICE_TOOLS,
            ):
                if event.content_delta is not None:
                    yield event.content_delta
                if event.completed_turn is not None:
                    completed_turn = event.completed_turn

            if completed_turn is None:
                raise LLMUnavailableError("Model stream ended without a completed turn")
            if not completed_turn.tool_calls:
                if completed_turn.content is None:
                    raise LLMUnavailableError("Model service returned an empty response")
                return

            if tool_round == MAX_TOOL_ROUNDS:
                raise LLMUnavailableError("Model exceeded the tool call limit")

            working_messages.append(
                {
                    "role": "assistant",
                    "content": completed_turn.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in completed_turn.tool_calls
                    ],
                }
            )

            for tool_call in completed_turn.tool_calls:
                tool_result = await self._execute_tool_call(
                    tool_call,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    idempotency_key=idempotency_key,
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": tool_result,
                    }
                )

        raise LLMUnavailableError("Model exceeded the tool call limit")

    async def _execute_tool_call(
        self,
        tool_call: FunctionToolCall,
        *,
        conversation_id: int,
        customer_id: int,
        idempotency_key: str,
    ) -> str:
        started_at = perf_counter()
        tool_result = await self._order_tool_executor.execute(
            tool_call,
            conversation_id=conversation_id,
            customer_id=customer_id,
            idempotency_key=idempotency_key,
        )
        duration_ms = (perf_counter() - started_at) * 1000

        try:
            result = json.loads(tool_result)
        except json.JSONDecodeError:
            result = {}
        succeeded = result.get("ok") is True
        error_code = None if succeeded else result.get("error", {}).get("code", "invalid_result")
        logger.info(
            "agent_tool_finished conversation_id=%s tool_call_id=%s tool_name=%s "
            "status=%s error_code=%s duration_ms=%.2f",
            conversation_id,
            tool_call.id,
            tool_call.name,
            "success" if succeeded else "rejected",
            error_code or "none",
            duration_ms,
        )
        return tool_result
