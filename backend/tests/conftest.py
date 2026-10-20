from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import engine, get_db_session
from app.main import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def application() -> Iterator[FastAPI]:
    get_settings.cache_clear()
    app = create_app()
    yield app
    get_settings.cache_clear()


@pytest.fixture
def client(application: FastAPI) -> Iterator[TestClient]:
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    settings = get_settings()

    if settings.environment == "production":
        raise RuntimeError("Integration tests must not run against production")

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()
            session = AsyncSession(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            )

            try:
                yield session
            finally:
                await session.close()

                if transaction.is_active:
                    await transaction.rollback()
    finally:
        await engine.dispose()


@pytest.fixture
async def api_client(
    application: FastAPI,
    db_session: AsyncSession,
) -> AsyncIterator[AsyncClient]:
    async def override_get_db_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    application.dependency_overrides[get_db_session] = override_get_db_session

    transport = ASGITransport(app=application)

    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client
    finally:
        application.dependency_overrides.pop(
            get_db_session,
            None,
        )
