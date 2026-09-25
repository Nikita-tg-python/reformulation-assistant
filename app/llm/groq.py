"""Groq via the groq SDK (OpenAI-compatible chat completions)."""

import json
from typing import Any

import groq

from app.llm.base import (
    RETRY_STATUSES,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMResponse,
    Message,
    Retry,
    ToolCall,
    ToolSpec,
    parse_retry_after,
    with_retries,
)

TEMPERATURE = 0.2

# gpt-oss on Groq sometimes emits its reasoning as plain text instead of a tool call and
# Groq answers 400 output_parse_failed. It is random and fails fast, so a quick retry helps.
# Other 400s are real request errors and are not retried.
PARSE_FAILED_RETRY_S = 1.0


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(
        self,
        api_key: str | None,
        model: str | None,
        timeout_s: float = 30,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model = model
        # Less reasoning = fewer tokens against the per-minute quota and fewer parse failures.
        # Default "low" only for gpt-oss: other models accept different values (Qwen: none).
        if reasoning_effort is None and model and model.startswith("openai/gpt-oss"):
            reasoning_effort = "low"
        self._reasoning = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        # max_retries=0: the SDK would otherwise retry silently on its own; we use with_retries.
        self._client = (
            groq.AsyncGroq(api_key=api_key, timeout=timeout_s, max_retries=0) if api_key else None
        )

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        # json_object works on all Groq models; json_schema is limited to a few models.
        extra = {"response_format": {"type": "json_object"}} if json_output else {}
        choice = await self._create(messages, **extra)
        return choice.message.content or ""

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        choice = await self._create(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ],
            tool_choice="auto",
        )
        calls = []
        for tc in choice.message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"groq returned invalid tool arguments for {tc.function.name}",
                    "llm_bad_response",
                ) from exc
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
        text = choice.message.content if not calls else None
        message = Message(role="assistant", content=choice.message.content or "", tool_calls=calls)
        return LLMResponse(text=text, tool_calls=calls, message=message)

    async def _create(self, messages: list[Message], **kwargs: Any) -> Any:
        if self._client is None:
            raise LLMNotConfiguredError("GROQ_API_KEY is not set")
        if not self._model:
            raise LLMNotConfiguredError("GROQ_MODEL is not set")
        try:
            client, payload = self._client, [_to_dict(m) for m in messages]
            response = await with_retries(
                lambda: client.chat.completions.create(
                    model=self._model,
                    messages=payload,
                    temperature=TEMPERATURE,
                    **self._reasoning,
                    **kwargs,
                ),
                _retry_info,
            )
        except groq.RateLimitError as exc:
            raise LLMError(f"groq rate limit: {exc.message}", "llm_rate_limited", 503) from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"groq error {exc.status_code}: {exc.message}") from exc
        except groq.APIError as exc:  # connection errors, timeouts
            raise LLMError(f"groq request failed: {type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise LLMError("groq returned no choices", "llm_bad_response")
        return response.choices[0]


def _retry_info(exc: Exception) -> Retry | None:
    if not isinstance(exc, groq.APIStatusError):
        return None
    if exc.status_code in RETRY_STATUSES:
        retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
        return Retry(str(exc.status_code), retry_after)
    if exc.status_code == 400 and _error_code(exc) == "output_parse_failed":
        return Retry("400 output_parse_failed", PARSE_FAILED_RETRY_S)
    return None


def _error_code(exc: groq.APIStatusError) -> str | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    return error.get("code") if isinstance(error, dict) else None


def _to_dict(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "content": m.content,
        }
    message: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    "arguments": json.dumps(c.arguments, ensure_ascii=False),
                },
            }
            for c in m.tool_calls
        ]
    return message
