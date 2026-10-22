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

    db_host: str = "127.0.0.1"
    db_port: int = 3307
    db_name: str = "retail_agent"
    db_user: str = "retail_app"
    db_password: SecretStr = SecretStr("")
    db_echo: bool = False

    llm_api_base: str = "https://api.deepseek.com"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-flash"

    milvus_uri: str = str(DEFAULT_MILVUS_PATH)
    milvus_collection: str = "retail_knowledge"
    rag_min_score: float = Field(default=0.52, ge=0, le=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
