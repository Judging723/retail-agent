from app.core.config import Settings


def test_default_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.debug is False


def test_settings_with_monkeypatch(monkeypatch):
    monkeypatch.setenv("RETAIL_ENVIRONMENT", "test")
    monkeypatch.setenv("RETAIL_DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.environment == "test"
    assert settings.debug is True
