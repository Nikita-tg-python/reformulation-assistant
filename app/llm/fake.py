"""Scripted LLM for tests: no network, no keys, deterministic."""

from collections import deque
from collections.abc import Callable, Iterable

from app.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec

# A scripted reply: plain text, a list of tool calls, a full LLMResponse,
# or a function of the messages returning one of those.
Reply = str | list[ToolCall] | LLMResponse
Script = Reply | Callable[[list[Message]], Reply]


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies: Iterable[Script] = ()) -> None:
        self._replies = deque(replies)
        self.calls: list[list[Message]] = []  # every request, for assertions

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        reply = self._next(messages)
        if not isinstance(reply, str):
            raise AssertionError(
                f"FakeLLM.complete expected a str reply, got {type(reply).__name__}"
            )
        return reply

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        reply = self._next(messages)
        if isinstance(reply, LLMResponse):
            return reply
        if isinstance(reply, str):
            return LLMResponse(text=reply, tool_calls=[], message=Message("assistant", reply))
        return LLMResponse(
            text=None, tool_calls=reply, message=Message("assistant", tool_calls=reply)
        )

    def _next(self, messages: list[Message]) -> Reply:
        self.calls.append(list(messages))
        if not self._replies:
            raise AssertionError("FakeLLM: no scripted replies left")
        script = self._replies.popleft()
        return script(messages) if callable(script) else script
