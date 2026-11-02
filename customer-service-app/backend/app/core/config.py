from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MILVUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "knowledge-pipeline"
    / "data"
    / "vector_store"
    / "retail_agent.db"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RETAIL_",
        extra="ignore",
    )

    app_name: str = "Retail Agent Backend"
    service_name: str = "retail-agent-backend"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"

    jwt_secret: SecretStr = SecretStr("development-only-change-me-at-least-32-bytes")
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_access_token_minutes: int = Field(default=60, ge=5, le=1440)

    db_host: str = "127.0.0.1"
    db_port: int = 3307
    db_name: str = "retail_agent"
    db_user: str = "retail_app"
    db_password: SecretStr = SecretStr("")
    db_echo: bool = False

    llm_api_base: str = "https://api.deepseek.com"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-flash"
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)
    llm_max_retries: int = Field(default=1, ge=0, le=5)

    milvus_uri: str = str(DEFAULT_MILVUS_PATH)
    milvus_collection: str = "retail_knowledge"
    rag_min_score: float = Field(default=0.52, ge=0, le=1)
    rag_reranker_enabled: bool = True
    rag_reranker_candidates: int = Field(default=4, ge=1, le=50)
    rag_warmup_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
