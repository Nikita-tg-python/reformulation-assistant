"""Retry policy for LLM providers: at most 2 retries, pauses 2 s and 5 s.

Retryable: 429 and 503 on both providers, and 400 output_parse_failed on Groq.
"""

import groq
import httpx
import pytest
from google.genai import errors

from app.llm import gemini as gemini_module
from app.llm import groq as groq_module
from app.llm.base import MAX_RETRY_AFTER_S, Retry, parse_retry_after, with_retries
from app.llm.groq import GroqClient


class ProviderError(Exception):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        self.status, self.retry_after = status, retry_after


def classify(exc):
    """Test double for a provider classifier: 429 and 503 are retryable."""
    if isinstance(exc, ProviderError) and exc.status in {429, 503}:
        return Retry(str(exc.status), exc.retry_after)
    return None


def script(*outcomes):
    """A call that raises or returns the given outcomes in order; counts attempts."""
    queue = list(outcomes)

    async def call():
        call.attempts += 1
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    call.attempts = 0
    return call


@pytest.fixture
def pauses():
    recorded: list[float] = []

    async def sleep(seconds: float) -> None:
        recorded.append(seconds)

    sleep.recorded = recorded
    return sleep


async def test_retries_429_and_503_with_fixed_pauses(pauses):
    call = script(ProviderError(429), ProviderError(503), "ok")

    assert await with_retries(call, classify, sleep=pauses) == "ok"
    assert call.attempts == 3
    assert pauses.recorded == [2.0, 5.0]


async def test_gives_up_after_two_retries(pauses):
    call = script(ProviderError(503), ProviderError(503), ProviderError(503), "never")

    with pytest.raises(ProviderError):
        await with_retries(call, classify, sleep=pauses)
    assert call.attempts == 3
    assert pauses.recorded == [2.0, 5.0]


async def test_retry_after_replaces_the_fixed_pause_and_is_capped(pauses):
    call = script(ProviderError(429, retry_after=7), ProviderError(429, retry_after=120), "ok")

    await with_retries(call, classify, sleep=pauses)

    assert pauses.recorded == [7, MAX_RETRY_AFTER_S]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 504])
async def test_errors_the_classifier_rejects_are_not_retried(pauses, status):
    call = script(ProviderError(status), "never")

    with pytest.raises(ProviderError):
        await with_retries(call, classify, sleep=pauses)
    assert call.attempts == 1
    assert pauses.recorded == []


async def test_non_provider_errors_are_not_retried(pauses):
    call = script(ValueError("bug"), "never")

    with pytest.raises(ValueError, match="bug"):
        await with_retries(call, classify, sleep=pauses)
    assert call.attempts == 1


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("7", 7.0),
        ("1.5", 1.5),
        ("54s", 54.0),
        (None, None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
    ],
)
def test_parse_retry_after(value, seconds):
    assert parse_retry_after(value) == seconds


def _response(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "http://x"))


def _groq_error(cls, status: int, code: str | None = None, headers: dict | None = None):
    body = {"error": {"message": "x", "type": "invalid_request_error", "code": code}}
    return cls("x", response=_response(status, headers), body=body)


def test_groq_retries_rate_limit_and_overload_with_retry_after():
    rate_limited = _groq_error(groq.RateLimitError, 429, headers={"retry-after": "3"})
    overloaded = _groq_error(groq.InternalServerError, 503)

    assert groq_module._retry_info(rate_limited) == Retry("429", 3.0)
    assert groq_module._retry_info(overloaded) == Retry("503", None)


def test_groq_retries_output_parse_failed_quickly_but_no_other_400():
    parse_failed = _groq_error(groq.BadRequestError, 400, "output_parse_failed")
    other_400 = _groq_error(groq.BadRequestError, 400, "context_length_exceeded")

    assert groq_module._retry_info(parse_failed) == Retry(
        "400 output_parse_failed", groq_module.PARSE_FAILED_RETRY_S
    )
    assert groq_module._retry_info(other_400) is None


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (groq.AuthenticationError, 401),
        (groq.PermissionDeniedError, 403),
        (groq.NotFoundError, 404),
        (groq.InternalServerError, 500),
        (groq.InternalServerError, 504),
    ],
)
def test_groq_other_errors_are_not_retried(cls, status):
    assert groq_module._retry_info(_groq_error(cls, status)) is None


def test_groq_connection_errors_are_not_retried():
    error = groq.APIConnectionError(request=httpx.Request("POST", "http://x"))
    assert groq_module._retry_info(error) is None


def test_groq_reasoning_effort_low_by_default_only_for_gpt_oss():
    assert GroqClient("k", "openai/gpt-oss-120b")._reasoning == {"reasoning_effort": "low"}
    assert GroqClient("k", "qwen/qwen3.8-27b")._reasoning == {}
    explicit = GroqClient("k", "qwen/qwen3.8-27b", reasoning_effort="none")
    assert explicit._reasoning == {"reasoning_effort": "none"}


def test_groq_sdk_retries_are_disabled():
    # Otherwise the SDK retries silently on its own and stacks with with_retries.
    assert GroqClient("key", "model")._client.max_retries == 0


def test_gemini_retry_delay_is_read_from_header_or_body():
    body = {
        "error": {
            "code": 429,
            "message": "quota",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "54s"}],
        }
    }
    from_body = errors.APIError(429, body, response=_response(429))
    from_header = errors.APIError(
        503, {"error": {"message": "busy"}}, _response(503, {"retry-after": "4"})
    )

    not_retryable = errors.APIError(400, {"error": {"message": "bad"}}, _response(400))

    assert gemini_module._retry_info(from_body) == Retry("429", 54.0)
    assert gemini_module._retry_info(from_header) == Retry("503", 4.0)
    assert gemini_module._retry_info(not_retryable) is None
    assert gemini_module._retry_info(RuntimeError("x")) is None
