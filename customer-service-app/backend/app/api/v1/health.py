from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import DatabaseSession, get_knowledge_runtime
from app.core.config import Settings, get_settings

SettingDependency = Annotated[Settings, Depends(get_settings)]

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessComponents(BaseModel):
    database: str
    rag: str


class ReadinessResponse(BaseModel):
    status: str
    components: ReadinessComponents


@router.get("/health", response_model=HealthResponse)
def health_check(settings: SettingDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readiness_check(
    session: DatabaseSession,
    response: Response,
) -> ReadinessResponse:
    try:
        await session.execute(text("SELECT 1"))
        database_status = "ready"
    except SQLAlchemyError:
        database_status = "unavailable"

    runtime_status = get_knowledge_runtime().status
    rag_status = {
        "not_initialized": "lazy",
        "ready": "ready",
        "degraded": "degraded",
        "closed": "lazy",
    }[runtime_status]

    if database_status == "unavailable":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall_status = "unavailable"
    elif rag_status == "degraded":
        overall_status = "degraded"
    else:
        overall_status = "ready"

    return ReadinessResponse(
        status=overall_status,
        components=ReadinessComponents(
            database=database_status,
            rag=rag_status,
        ),
    )
