from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


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
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    db_host: str = "127.0.0.1"
    db_port: int = 3307
    db_name: str = "retail_agent"
    db_user: str = "retail_app"
    db_password: SecretStr = SecretStr("")
    db_echo: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
