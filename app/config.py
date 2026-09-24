from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (see .env.example)."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    database_url: str = "postgresql://postgres:postgres@localhost:5433/reformulation"

    llm_provider: Literal["gemini", "groq"] = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    # Deviation from spec (all-MiniLM-L6-v2): the corpus is Ukrainian, see .env.example.
    embedding_model: str = "intfloat/multilingual-e5-small"

    agent_max_iterations: int = Field(default=6, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60, gt=0, le=600)


@lru_cache
def get_settings() -> Settings:
    return Settings()
