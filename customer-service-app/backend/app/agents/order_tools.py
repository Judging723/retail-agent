import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm import FunctionToolCall, ToolDefinition
from app.models import Order, ToolAuditLog, ToolAuditStatus

GET_ORDER_STATUS_TOOL: ToolDefinition = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "查询当前客户自己的订单、物流和退款状态。"
            "当用户询问具体订单且已经提供订单号时调用；不要猜测订单号。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_no": {
                    "type": "string",
                    "description": "用户提供的完整订单号",
                }
            },
            "required": ["order_no"],
            "additionalProperties": False,
        },
    },
}


class GetOrderStatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class OrderToolExecutor:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def execute(
        self,
        tool_call: FunctionToolCall,
        *,
        conversation_id: int,
        customer_id: int,
    ) -> str:
        if tool_call.name != "get_order_status":
            payload = {
                "ok": False,
                "error": {
                    "code": "unknown_tool",
                    "message": "不支持该工具。",
                },
            }
            await self._record_rejection(
                tool_call=tool_call,
                conversation_id=conversation_id,
                customer_id=customer_id,
                arguments={"raw": tool_call.arguments},
                error_code="unknown_tool",
                result=payload,
            )
            return self._serialize(payload)

        try:
            arguments = GetOrderStatusArguments.model_validate_json(tool_call.arguments)
        except ValidationError:
            payload = {
                "ok": False,
                "error": {
                    "code": "invalid_arguments",
                    "message": "订单号参数格式无效，请向用户确认完整订单号。",
                },
            }
            await self._record_rejection(
                tool_call=tool_call,
                conversation_id=conversation_id,
                customer_id=customer_id,
                arguments={"raw": tool_call.arguments},
                error_code="invalid_arguments",
                result=payload,
            )
            return self._serialize(payload)

        async with self._session.begin():
            statement = (
                select(Order)
                .where(
                    Order.order_no == arguments.order_no,
                    Order.customer_id == customer_id,
                )
                .options(
                    selectinload(Order.shipment),
                    selectinload(Order.refunds),
                )
            )
            order = await self._session.scalar(statement)

            if order is None:
                payload = {
                    "ok": False,
                    "error": {
                        "code": "order_not_found",
                        "message": "当前客户名下未找到该订单，请核对订单号。",
                    },
                }
                audit_status = ToolAuditStatus.REJECTED
                error_code = "order_not_found"
            else:
                payload = {
                    "ok": True,
                    "order": {
                        "order_no": order.order_no,
                        "status": order.status.value,
                        "total_amount": str(order.total_amount),
                        "currency": order.currency,
                        "shipment": (
                            {
                                "carrier": order.shipment.carrier,
                                "tracking_no": order.shipment.tracking_no,
                                "status": order.shipment.status.value,
                                "latest_event": order.shipment.latest_event,
                            }
                            if order.shipment is not None
                            else None
                        ),
                        "refunds": [
                            {
                                "refund_no": refund.refund_no,
                                "amount": str(refund.amount),
                                "status": refund.status.value,
                            }
                            for refund in order.refunds
                        ],
                    },
                }
                audit_status = ToolAuditStatus.SUCCESS
                error_code = None

            self._session.add(
                ToolAuditLog(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=arguments.model_dump(mode="json"),
                    status=audit_status,
                    result=payload,
                    error_code=error_code,
                )
            )

        return self._serialize(payload)

    async def _record_rejection(
        self,
        *,
        tool_call: FunctionToolCall,
        conversation_id: int,
        customer_id: int,
        arguments: dict[str, Any],
        error_code: str,
        result: dict[str, Any],
    ) -> None:
        async with self._session.begin():
            self._session.add(
                ToolAuditLog(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=arguments,
                    status=ToolAuditStatus.REJECTED,
                    result=result,
                    error_code=error_code,
                )
            )

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
