import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import get_knowledge_runtime
from app.api.router import api_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.exception_handlers import register_exception_handlers
from app.middleware.request_logging import register_request_logging

logger = logging.getLogger("app.lifecycle")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    runtime = get_knowledge_runtime()

    if settings.rag_warmup_enabled and settings.environment != "test":
        ready = await asyncio.to_thread(runtime.warmup)
        logger.info("application_startup rag_status=%s", "ready" if ready else "degraded")
    else:
        logger.info("application_startup rag_status=lazy")

    try:
        yield
    finally:
        await asyncio.to_thread(runtime.close)
        await engine.dispose()
        get_knowledge_runtime.cache_clear()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description=settings.service_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    register_request_logging(application)
    register_exception_handlers(application)

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )
    return application


app = create_app()
