from app.config import Settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient


def create_llm_client(settings: Settings) -> LLMClient:
    """Pick the provider from LLM_PROVIDER. Switching providers is one env variable."""

    def secret(value):
        return value.get_secret_value() if value else None

    if settings.llm_provider == "gemini":
        return GeminiClient(secret(settings.gemini_api_key), settings.gemini_model)
    if settings.llm_provider == "groq":
        return GroqClient(
            secret(settings.groq_api_key),
            settings.groq_model,
            reasoning_effort=settings.groq_reasoning_effort,
        )
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
