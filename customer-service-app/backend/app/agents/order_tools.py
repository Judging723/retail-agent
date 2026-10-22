import json
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.llm import FunctionToolCall, ToolDefinition
from app.models import (
    Order,
    OrderStatus,
    Refund,
    RefundStatus,
    ToolAuditLog,
    ToolAuditStatus,
)

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

REQUEST_REFUND_TOOL: ToolDefinition = {
    "type": "function",
    "function": {
        "name": "request_refund",
        "description": (
            "为当前客户自己的订单提交退款申请。"
            "只有当用户明确要求提交退款、并提供订单号、金额和原因时才能调用；"
            "如果用户只是咨询退款政策或信息不完整，必须先回答或追问。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_no": {
                    "type": "string",
                    "description": "用户提供的完整订单号",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "退款金额，最多两位小数",
                },
                "reason": {
                    "type": "string",
                    "description": "用户说明的退款原因",
                },
            },
            "required": ["order_no", "amount", "reason"],
            "additionalProperties": False,
        },
    },
}

CUSTOMER_SERVICE_TOOLS = [GET_ORDER_STATUS_TOOL, REQUEST_REFUND_TOOL]

REFUNDABLE_ORDER_STATUSES = {
    OrderStatus.PAID,
    OrderStatus.FULFILLING,
    OrderStatus.SHIPPED,
    OrderStatus.COMPLETED,
}
RESERVED_REFUND_STATUSES = {
    RefundStatus.REQUESTED,
    RefundStatus.PROCESSING,
    RefundStatus.SUCCEEDED,
}


class GetOrderStatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class RequestRefundArguments(GetOrderStatusArguments):
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    reason: str = Field(min_length=2, max_length=255)


class OrderToolExecutor:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def execute(
        self,
        tool_call: FunctionToolCall,
        *,
        conversation_id: int,
        customer_id: int,
        idempotency_key: str,
    ) -> str:
        if tool_call.name == "get_order_status":
            try:
                arguments = GetOrderStatusArguments.model_validate_json(tool_call.arguments)
            except ValidationError:
                return await self._reject_invalid_arguments(
                    tool_call=tool_call,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    message="订单号参数格式无效，请向用户确认完整订单号。",
                )
            return await self._get_order_status(
                tool_call=tool_call,
                arguments=arguments,
                conversation_id=conversation_id,
                customer_id=customer_id,
            )

        if tool_call.name == "request_refund":
            try:
                arguments = RequestRefundArguments.model_validate_json(tool_call.arguments)
            except ValidationError:
                return await self._reject_invalid_arguments(
                    tool_call=tool_call,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    message="退款参数无效，请确认订单号、金额和退款原因。",
                )
            return await self._request_refund(
                tool_call=tool_call,
                arguments=arguments,
                conversation_id=conversation_id,
                customer_id=customer_id,
                idempotency_key=idempotency_key,
            )

        payload = self._error_payload("unknown_tool", "不支持该工具。")
        await self._record_rejection(
            tool_call=tool_call,
            conversation_id=conversation_id,
            customer_id=customer_id,
            arguments={"raw": tool_call.arguments},
            error_code="unknown_tool",
            result=payload,
        )
        return self._serialize(payload)

    async def _get_order_status(
        self,
        *,
        tool_call: FunctionToolCall,
        arguments: GetOrderStatusArguments,
        conversation_id: int,
        customer_id: int,
    ) -> str:
        async with self._session.begin():
            order = await self._find_customer_order(arguments.order_no, customer_id)

            if order is None:
                payload = self._error_payload(
                    "order_not_found",
                    "当前客户名下未找到该订单，请核对订单号。",
                )
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

            self._add_audit(
                tool_call=tool_call,
                conversation_id=conversation_id,
                customer_id=customer_id,
                arguments=arguments.model_dump(mode="json"),
                status=audit_status,
                result=payload,
                error_code=error_code,
            )

        return self._serialize(payload)

    async def _request_refund(
        self,
        *,
        tool_call: FunctionToolCall,
        arguments: RequestRefundArguments,
        conversation_id: int,
        customer_id: int,
        idempotency_key: str,
    ) -> str:
        async with self._session.begin():
            existing_refund = await self._session.scalar(
                select(Refund)
                .where(Refund.idempotency_key == idempotency_key)
                .options(selectinload(Refund.order))
            )
            if existing_refund is not None:
                return self._handle_refund_replay(
                    tool_call=tool_call,
                    arguments=arguments,
                    existing_refund=existing_refund,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                )

            order = await self._find_customer_order(arguments.order_no, customer_id)
            if order is None:
                return self._reject_refund(
                    tool_call=tool_call,
                    arguments=arguments,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    code="order_not_found",
                    message="当前客户名下未找到该订单，请核对订单号。",
                )

            if order.status not in REFUNDABLE_ORDER_STATUSES:
                return self._reject_refund(
                    tool_call=tool_call,
                    arguments=arguments,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    code="refund_not_allowed",
                    message=f"订单状态 {order.status.value} 暂不允许提交退款申请。",
                )

            reserved_amount = sum(
                (
                    refund.amount
                    for refund in order.refunds
                    if refund.status in RESERVED_REFUND_STATUSES
                ),
                start=Decimal("0.00"),
            )
            refundable_amount = order.total_amount - reserved_amount
            if arguments.amount > refundable_amount:
                return self._reject_refund(
                    tool_call=tool_call,
                    arguments=arguments,
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    code="amount_exceeds_refundable",
                    message=f"本订单当前最多可申请退款 {refundable_amount:.2f} {order.currency}。",
                )

            refund = Refund(
                refund_no=self._generate_refund_no(),
                idempotency_key=idempotency_key,
                order_id=order.id,
                amount=arguments.amount,
                reason=arguments.reason,
                status=RefundStatus.REQUESTED,
            )
            self._session.add(refund)
            await self._session.flush()

            payload = self._refund_success_payload(
                refund=refund,
                order=order,
                idempotent_replay=False,
            )
            self._add_audit(
                tool_call=tool_call,
                conversation_id=conversation_id,
                customer_id=customer_id,
                arguments=arguments.model_dump(mode="json"),
                status=ToolAuditStatus.SUCCESS,
                result=payload,
                error_code=None,
            )

        return self._serialize(payload)

    def _handle_refund_replay(
        self,
        *,
        tool_call: FunctionToolCall,
        arguments: RequestRefundArguments,
        existing_refund: Refund,
        conversation_id: int,
        customer_id: int,
    ) -> str:
        order = existing_refund.order
        same_request = (
            order.customer_id == customer_id
            and order.order_no == arguments.order_no
            and existing_refund.amount == arguments.amount
            and existing_refund.reason == arguments.reason
        )
        if not same_request:
            return self._reject_refund(
                tool_call=tool_call,
                arguments=arguments,
                conversation_id=conversation_id,
                customer_id=customer_id,
                code="idempotency_conflict",
                message="同一请求不能提交不同的退款内容。",
            )

        payload = self._refund_success_payload(
            refund=existing_refund,
            order=order,
            idempotent_replay=True,
        )
        self._add_audit(
            tool_call=tool_call,
            conversation_id=conversation_id,
            customer_id=customer_id,
            arguments=arguments.model_dump(mode="json"),
            status=ToolAuditStatus.SUCCESS,
            result=payload,
            error_code=None,
        )
        return self._serialize(payload)

    def _reject_refund(
        self,
        *,
        tool_call: FunctionToolCall,
        arguments: RequestRefundArguments,
        conversation_id: int,
        customer_id: int,
        code: str,
        message: str,
    ) -> str:
        payload = self._error_payload(code, message)
        self._add_audit(
            tool_call=tool_call,
            conversation_id=conversation_id,
            customer_id=customer_id,
            arguments=arguments.model_dump(mode="json"),
            status=ToolAuditStatus.REJECTED,
            result=payload,
            error_code=code,
        )
        return self._serialize(payload)

    async def _find_customer_order(self, order_no: str, customer_id: int) -> Order | None:
        return await self._session.scalar(
            select(Order)
            .where(
                Order.order_no == order_no,
                Order.customer_id == customer_id,
            )
            .options(
                selectinload(Order.shipment),
                selectinload(Order.refunds),
            )
        )

    async def _reject_invalid_arguments(
        self,
        *,
        tool_call: FunctionToolCall,
        conversation_id: int,
        customer_id: int,
        message: str,
    ) -> str:
        payload = self._error_payload("invalid_arguments", message)
        await self._record_rejection(
            tool_call=tool_call,
            conversation_id=conversation_id,
            customer_id=customer_id,
            arguments={"raw": tool_call.arguments},
            error_code="invalid_arguments",
            result=payload,
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
            self._add_audit(
                tool_call=tool_call,
                conversation_id=conversation_id,
                customer_id=customer_id,
                arguments=arguments,
                status=ToolAuditStatus.REJECTED,
                result=result,
                error_code=error_code,
            )

    def _add_audit(
        self,
        *,
        tool_call: FunctionToolCall,
        conversation_id: int,
        customer_id: int,
        arguments: dict[str, Any],
        status: ToolAuditStatus,
        result: dict[str, Any],
        error_code: str | None,
    ) -> None:
        self._session.add(
            ToolAuditLog(
                conversation_id=conversation_id,
                customer_id=customer_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=arguments,
                status=status,
                result=result,
                error_code=error_code,
            )
        )

    @staticmethod
    def _refund_success_payload(
        *,
        refund: Refund,
        order: Order,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "refund": {
                "refund_no": refund.refund_no,
                "order_no": order.order_no,
                "amount": str(refund.amount),
                "currency": order.currency,
                "status": refund.status.value,
                "idempotent_replay": idempotent_replay,
            },
        }

    @staticmethod
    def _error_payload(code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "error": {"code": code, "message": message}}

    @staticmethod
    def _generate_refund_no() -> str:
        return f"REFUND-{uuid4().hex[:20].upper()}"

    @staticmethod
    def _serialize(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
