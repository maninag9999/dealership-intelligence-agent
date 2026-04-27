"""
Runtime configuration for the Dealership Intelligence Agent.

All settings are loaded from environment variables / .env file via
pydantic-settings.  Sensitive values (passwords, API keys) are typed as
``SecretStr`` so they are redacted in repr() / logging output.

Usage::

    from dealership.common.config import get_settings

    settings = get_settings()
    db_url = settings.postgres_dsn()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dealership"
    postgres_user: str = "dealership_user"
    postgres_password: SecretStr = SecretStr("changeme")

    # DuckDB
    duckdb_path: str = "data/warehouse/dealership.duckdb"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "dealership_docs"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    # Groq
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = "llama-3.3-70b-versatile"

    # Synthetic data
    synthetic_seed: int = 42
    synthetic_num_reps: int = 20
    synthetic_num_customers: int = 800
    synthetic_num_vehicles: int = 450
    synthetic_num_sales: int = 1_400
    synthetic_date_start: str = "2022-01-01"
    synthetic_date_end: str = "2023-12-31"

    # Paths
    data_raw_path: str = "data/raw/synthetic"
    data_warehouse_path: str = "data/warehouse"
    data_debug_path: str = "data/debug/synthetic"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/dealership.log"

    @field_validator("postgres_port", "chroma_port", mode="before")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not (1 <= int(v) <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return int(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_upper(cls, v: str) -> str:
        v = v.upper()
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v

    @model_validator(mode="after")
    def _ensure_paths_exist(self) -> Settings:
        for attr in ("data_raw_path", "data_warehouse_path", "data_debug_path"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        return self

    def postgres_dsn(self, *, driver: str = "postgresql+psycopg2") -> str:
        """Return SQLAlchemy-compatible DSN (unwraps secret)."""
        pwd = self.postgres_password.get_secret_value()
        return f"{driver}://{self.postgres_user}:{pwd}" f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    def duckdb_file(self) -> Path:
        return Path(self.duckdb_path)

    def raw_path(self) -> Path:
        return Path(self.data_raw_path)

    def debug_path(self) -> Path:
        return Path(self.data_debug_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
