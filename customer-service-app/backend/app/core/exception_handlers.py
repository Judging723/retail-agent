from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


async def internal_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or str(uuid4())

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": "服务器内部错误",
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(
        Exception,
        internal_exception_handler,
    )
