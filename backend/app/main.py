from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.middleware.request_logging import request_logging_middleware

from app.core.logging import configure_logging

settings = get_settings()

configure_logging(settings.log_level)


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        description=settings.service_name,
        version=settings.app_version,
    )

    application.middleware("http")(request_logging_middleware)

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )
    return application


app = create_app()
