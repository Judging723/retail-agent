from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings

settings = get_settings()


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        description=settings.service_name,
        version=settings.app_version,
    )

    application.include_router(
        api_router,
        prefix=settings.api_v1_prefix,
    )
    return application


app = create_app()
