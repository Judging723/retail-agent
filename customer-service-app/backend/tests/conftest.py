from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_knowledge_retriever
from app.core.config import get_settings
from app.core.database import engine, get_db_session
from app.main import create_app
from app.rag import RetrievedDocument


class StubKnowledgeRetriever:
    def search(self, query: str, *, limit: int = 2) -> list[RetrievedDocument]:
        if not query.strip() or limit <= 0:
            return []
        return [
            RetrievedDocument(
                source_id="car-test-shopping-case",
                title="购物澄清案例：测试商品",
                category="购物歧义与澄清",
                content="顾客需求不完整时，客服通过询问类型、规格和使用场景进行澄清。",
                revision="3fe3a4223357",
                source_date="2024-04-21",
                source_url="https://github.com/ygan/Chinese-Ambiguous-Reference",
                license="MIT",
                score=0.9,
            )
        ][:limit]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def application() -> Iterator[FastAPI]:
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_knowledge_retriever] = StubKnowledgeRetriever
    yield app
    app.dependency_overrides.pop(get_knowledge_retriever, None)
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
