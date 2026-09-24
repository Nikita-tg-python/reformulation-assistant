"""Provider-neutral LLM interface. Everything outside app/llm/ talks only to LLMClient."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from app.errors import AppError

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
