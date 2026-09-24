"""Gemini via the google-genai SDK (generate_content API, manual function calling)."""

from google import genai
from google.genai import errors, types

from app.llm.base import (
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
)

TEMPERATURE = 0.2


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str | None, model: str | None, timeout_s: float = 30) -> None:
        self._model = model
        # Build the SDK client only with a key: the app must start (and /health work) without one.
        self._client = (
            genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
            )
            if api_key
            else None
        )

    async def complete(self, messages: list[Message], *, json_output: bool = False) -> str:
        config = self._config(messages)
        if json_output:
            config.response_mime_type = "application/json"
        response = await self._generate(messages, config)
        return _text(response)

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        config = self._config(messages)
        config.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name, description=t.description, parameters_json_schema=t.parameters
                    )
                    for t in tools
                ]
            )
        ]
        # We run our own loop (limits, trace), so the SDK must not execute functions itself.
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        response = await self._generate(messages, config)

        calls = [
            ToolCall(id=fc.id or f"call_{i}", name=fc.name or "", arguments=dict(fc.args or {}))
            for i, fc in enumerate(response.function_calls or [])
        ]
        text = _text(response) if not calls else None
        message = Message(
            role="assistant",
            content=text or "",
            tool_calls=calls,
            provider_data=response.candidates[0].content,
        )
        return LLMResponse(text=text, tool_calls=calls, message=message)

    def _config(self, messages: list[Message]) -> types.GenerateContentConfig:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        return types.GenerateContentConfig(
            system_instruction=system or None, temperature=TEMPERATURE
        )

    async def _generate(
        self, messages: list[Message], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        if self._client is None:
            raise LLMNotConfiguredError("GEMINI_API_KEY is not set")
        if not self._model:
            raise LLMNotConfiguredError("GEMINI_MODEL is not set")
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=_to_contents(messages), config=config
            )
        except errors.APIError as exc:
            if exc.code == 429:
                raise LLMError(
                    f"gemini rate limit: {exc.message}", "llm_rate_limited", 503
                ) from exc
            raise LLMError(f"gemini error {exc.code}: {exc.message}") from exc
        except Exception as exc:  # network, timeout, SDK-level errors
            raise LLMError(f"gemini request failed: {type(exc).__name__}: {exc}") from exc
        if not response.candidates or response.candidates[0].content is None:
            reason = (
                response.candidates[0].finish_reason if response.candidates else "no candidates"
            )
            raise LLMError(f"gemini returned no content ({reason})", "llm_bad_response")
        return response


def _text(response: types.GenerateContentResponse) -> str:
    parts = response.candidates[0].content.parts or []
    return "".join(p.text for p in parts if p.text and not p.thought)


def _to_contents(messages: list[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for m in messages:
        if m.role == "system":
            continue  # goes to system_instruction
        if m.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=m.content)])
            )
        elif m.role == "assistant":
            if isinstance(m.provider_data, types.Content):
                contents.append(m.provider_data)  # verbatim: keeps thought signatures
            else:
                parts = [types.Part.from_text(text=m.content)] if m.content else []
                parts += [
                    types.Part(function_call=types.FunctionCall(name=c.name, args=c.arguments))
                    for c in m.tool_calls
                ]
                contents.append(types.Content(role="model", parts=parts))
        elif m.role == "tool":
            part = types.Part.from_function_response(
                name=m.name or "", response={"result": m.content}
            )
            # The API takes function results in a "user" turn ("tool" is rejected with 400).
            # Results of parallel calls go into one turn, in call order.
            last = contents[-1] if contents else None
            if last and last.role == "user" and all(p.function_response for p in last.parts):
                last.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
    return contents
