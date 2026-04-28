from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT_DIR / "db" / "schema.sql"
SEEDS_PATH = ROOT_DIR / "seeds" / "universities.json"
SCREENSHOT_DIR = Path("/tmp/screenshots")


class Settings(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    database_url: str = Field(
        default="sqlite+aio:///./data/university.db",
        alias="DATABASE_URL",
    )
    model_provider: str = Field(default="anthropic", alias="MODEL_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    llm_extraction_model: str = Field(
        default="claude-haiku-4-5",
        alias="LLM_EXTRACTION_MODEL",
    )
    client_model: str = Field(default="claude-sonnet-4-5", alias="CLIENT_MODEL")
    scrape_delay_min: float = Field(default=3.0, alias="SCRAPE_DELAY_MIN", ge=0.0)
    scrape_delay_max: float = Field(default=9.0, alias="SCRAPE_DELAY_MAX", ge=0.0)
    extraction_cache_ttl: int = Field(
        default=86400,
        alias="EXTRACTION_CACHE_TTL",
        ge=0,
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    @field_validator("model_provider")
    @classmethod
    def validate_model_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"anthropic", "openai", "openrouter"}:
            raise ValueError("MODEL_PROVIDER must be 'anthropic', 'openai', or 'openrouter'")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be a valid Python logging level")
        return normalized

    @field_validator("openai_base_url")
    @classmethod
    def validate_openai_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("OPENAI_BASE_URL must start with http:// or https://")
        return normalized

    @field_validator("openrouter_base_url")
    @classmethod
    def validate_openrouter_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("OPENROUTER_BASE_URL must start with http:// or https://")
        return normalized

    @property
    def sqlite_path(self) -> str:
        prefix = "sqlite+aio:///"
        if not self.database_url.startswith(prefix):
            raise ValueError("DATABASE_URL must start with sqlite+aio:///")
        raw_path = self.database_url.removeprefix(prefix)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return str(path.resolve())

    @property
    def data_dir(self) -> Path:
        return Path(self.sqlite_path).parent

    @property
    def openai_compatible_api_key(self) -> str:
        if self.model_provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key

    @property
    def openai_compatible_base_url(self) -> str:
        if self.model_provider == "openrouter":
            return self.openrouter_base_url
        return self.openai_base_url


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


settings = Settings()
configure_logging(settings.log_level)