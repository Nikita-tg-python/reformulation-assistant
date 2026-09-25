"""Provider-neutral LLM interface. Everything outside app/llm/ talks only to LLMClient."""

import asyncio
import logging
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.errors import AppError

logger = logging.getLogger(__name__)

# Retry policy for provider calls. Providers decide which errors are retryable (see
# `Retry`): rate limit (429), overload (503) and, on Groq, 400 output_parse_failed.
RETRY_STATUSES = frozenset({429, 503})
RETRY_DELAYS_S = (2.0, 5.0)  # one pause per retry: at most 2 retries
MAX_RETRY_AFTER_S = 30.0  # a longer wait asked by the provider: give up at once, no retry

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # role="assistant"
    tool_call_id: str | None = None  # role="tool"
    name: str | None = None  # role="tool": name of the tool that produced this result
    # Provider-native form of an assistant turn, replayed verbatim by the same provider.
    # Gemini needs this: its function-call parts carry thought signatures we must not drop.
    provider_data: Any = None


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    message: Message  # the assistant turn, ready to append to the history


class LLMError(AppError):
    """Provider failure surfaced to the API in the common error format."""

    def __init__(self, message: str, code: str = "llm_error", status_code: int = 502) -> None:
        super().__init__(status_code, code, message)


class LLMNotConfiguredError(LLMError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="llm_not_configured", status_code=503)


class LLMClient(ABC):
    provider: str

    @abstractmethod
    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        """Single completion without tools. json_output=True asks the provider for a JSON object."""

    @abstractmethod
    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        """One model turn that may request tool calls. The caller runs the tools and loops."""


@dataclass(frozen=True)
class Retry:
    """A provider error worth retrying."""

    reason: str  # for the log, e.g. "429" or "400 output_parse_failed"
    after_s: float | None = None  # wait asked by the provider; None -> RETRY_DELAYS_S


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    classify: Callable[[Exception], Retry | None],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `call`, retrying up to len(RETRY_DELAYS_S) times when `classify(exc)` says so.

    `classify` returns a `Retry` for a retryable provider error and None for anything
    else; non-retryable errors are raised at once, the last attempt's error propagates.
    A provider asking to wait longer than MAX_RETRY_AFTER_S (e.g. Groq's daily token limit:
    "try again in 12m54s") is not retried: waiting 30 s and failing again would only burn
    the agent's time budget and turn an honest 503 rate limit into a 504 timeout.
    """
    for attempt, delay in enumerate(RETRY_DELAYS_S):
        try:
            return await call()
        except Exception as exc:
            retry = classify(exc)
            if retry is None or (retry.after_s or 0) > MAX_RETRY_AFTER_S:
                raise
            wait = retry.after_s if retry.after_s is not None else delay
            logger.warning(
                "llm provider returned %s, retry %d/%d in %.1f s",
                retry.reason,
                attempt + 1,
                len(RETRY_DELAYS_S),
                wait,
            )
            await sleep(wait)
    return await call()


def parse_retry_after(value: object) -> float | None:
    """Seconds from a Retry-After value ("7", "1.5", "54s"); None if absent or not seconds."""
    if value is None:
        return None
    try:
        seconds = float(str(value).strip().removesuffix("s"))
    except ValueError:
        return None  # HTTP-date form: fall back to the fixed pause
    return seconds if seconds >= 0 else None


def api_key_problem(env_name: str, key: str | None) -> str | None:
    """A key with a stray non-ASCII or blank character breaks every HTTP request (headers are
    ASCII) with an opaque encoding error; report it clearly instead."""
    for position, char in enumerate(key or ""):
        if not char.isascii() or not char.isprintable() or char.isspace():
            name = unicodedata.name(char, repr(char))
            return (
                f"{env_name} contains an invalid character at position {position} ({name}): "
                "check the keyboard layout and stray spaces"
            )
    return None
