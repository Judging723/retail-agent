from app.core.config import Settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.delenv("RETAIL_ENVIRONMENT", raising=False)
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.llm_api_base == "https://api.deepseek.com"
    assert settings.llm_model == "deepseek-flash"
    assert settings.llm_api_key.get_secret_value() == ""
    assert settings.rag_min_score == 0.52
    assert settings.rag_reranker_enabled is True
    assert settings.rag_reranker_candidates == 8


def test_llm_settings_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("RETAIL_LLM_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("RETAIL_LLM_API_KEY", "local-key")
    monkeypatch.setenv("RETAIL_LLM_MODEL", "local-model")

    settings = Settings(_env_file=None)

    assert settings.llm_api_base == "http://localhost:8000/v1"
    assert settings.llm_api_key.get_secret_value() == "local-key"
    assert settings.llm_model == "local-model"
