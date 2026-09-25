from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SearchMode = Literal["vector", "hybrid"]


class Settings(BaseSettings):
    """All configuration comes from environment variables (see .env.example)."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    database_url: str = "postgresql://postgres:postgres@localhost:5433/reformulation"

    # groq by default: on free tiers it was the stable one (see README, live agent state).
    llm_provider: Literal["gemini", "groq"] = "groq"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None
    # None = automatic: "low" for openai/gpt-oss-*, not sent for other models.
    groq_reasoning_effort: Literal["none", "default", "low", "medium", "high"] | None = None

    # Deviation from spec (all-MiniLM-L6-v2): the corpus is Ukrainian, see .env.example.
    embedding_model: str = "intfloat/multilingual-e5-small"
    # hybrid: vector + full-text search fused with RRF; vector: embeddings only.
    search_mode: SearchMode = "hybrid"

    # loop: free tool calling (up to agent_max_iterations LLM calls);
    # pipeline: fixed tool order chosen by code, 2 LLM calls (3 with a retry).
    agent_mode: Literal["loop", "pipeline"] = "pipeline"
    agent_max_iterations: int = Field(default=6, ge=1, le=20)
    agent_timeout_seconds: float = Field(default=60, gt=0, le=600)


@lru_cache
def get_settings() -> Settings:
    return Settings()
