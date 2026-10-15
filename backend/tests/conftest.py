from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    get_settings.cache_clear()
    application = create_app()

    with TestClient(application) as test_client:
        yield test_client

    get_settings.cache_clear()
