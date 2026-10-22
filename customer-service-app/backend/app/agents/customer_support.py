from app.agents.order_tools import CUSTOMER_SERVICE_TOOLS, OrderToolExecutor
from app.llm import ChatMessage, LLMClient, LLMUnavailableError

MAX_TOOL_ROUNDS = 3


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
                tool_result = await self._order_tool_executor.execute(
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
