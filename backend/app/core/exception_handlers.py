from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppException
from app.services.exceptions import (
    ConversationClosedError,
    ConversationNotFoundError,
)


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    request_id = get_request_id(request)
    content = {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
            "details": exc.details,
        }
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )


async def conversation_not_found_exception_handler(
    request: Request,
    exc: ConversationNotFoundError,
) -> JSONResponse:
    return await app_exception_handler(
        request,
        AppException(
            code="conversation_not_found",
            message=str(exc),
            status_code=status.HTTP_404_NOT_FOUND,
            details={
                "conversation_no": exc.conversation_no,
            },
        ),
    )


async def conversation_closed_exception_handler(
    request: Request,
    exc: ConversationClosedError,
) -> JSONResponse:
    return await app_exception_handler(
        request,
        AppException(
            code="conversation_closed",
            message=str(exc),
            status_code=status.HTTP_409_CONFLICT,
            details={
                "conversation_no": exc.conversation_no,
            },
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = get_request_id(request)
    if exc.status_code == 404:
        code = "not_found"
    else:
        code = "http_error"
    content = {
        "error": {
            "code": code,
            "message": exc.detail,
            "request_id": request_id,
            "details": None,
        }
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers={"X-Request-ID": request_id},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = get_request_id(request)
    content = {
        "error": {
            "code": "validation_error",
            "message": "Input validation failed",
            "request_id": request_id,
            "details": jsonable_encoder(exc.errors()),
        }
    }
    return JSONResponse(
        status_code=422,
        content=content,
        headers={"X-Request-ID": request_id},
    )


async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = get_request_id(request)
    content = {
        "error": {
            "code": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "details": None,
        }
    }
    return JSONResponse(
        status_code=500,
        content=content,
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(AppException, app_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(
        ConversationNotFoundError,
        conversation_not_found_exception_handler,
    )
    application.add_exception_handler(
        ConversationClosedError,
        conversation_closed_exception_handler,
    )
    application.add_exception_handler(Exception, internal_exception_handler)
