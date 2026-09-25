"""Reformulation agent: a plain tool-calling loop, no frameworks.

One iteration = one LLM call. The loop stops with a validated answer, or raises
AgentError carrying the trace collected so far (agent_timeout on limits).
"""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from pydantic import ValidationError

from app.agent import prompts
from app.agent.tools import NUTRIENTS, TOOL_SPECS, compact_result
from app.errors import AppError
from app.llm.base import LLMClient, LLMError, Message, ToolCall
from app.schemas import NutritionValues, ReformulateRequest, ReformulationDraft

Trace = list[dict[str, Any]]

logger = logging.getLogger(__name__)


class ToolExecutor(Protocol):
    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AgentError(AppError):
    def __init__(self, status_code: int, code: str, message: str, trace: Trace) -> None:
        super().__init__(status_code, code, message, extra={"trace": trace})
        self.trace = trace


class AgentTimeoutError(AgentError):
    def __init__(self, message: str, trace: Trace) -> None:
        super().__init__(504, "agent_timeout", message, trace)


@dataclass
class AgentResult:
    answer: ReformulationDraft
    trace: Trace
    iterations: int


async def run_agent(
    llm: LLMClient,
    tools: ToolExecutor,
    request: ReformulateRequest,
    *,
    max_iterations: int,
    timeout_s: float,
) -> AgentResult:
    trace: Trace = []  # shared with _loop, so a timeout still returns what was collected
    return await guarded(_loop(llm, tools, request, trace, max_iterations), trace, timeout_s)


async def guarded(work: Awaitable[AgentResult], trace: Trace, timeout_s: float) -> AgentResult:
    """Run an agent body under the time limit; every failure becomes AgentError + trace.

    Shared by the loop and the fixed pipeline so both fail the same way.
    """
    try:
        async with asyncio.timeout(timeout_s) as deadline:
            return await work
    except TimeoutError:
        if deadline.expired():
            raise AgentTimeoutError(f"agent exceeded {timeout_s:g} s", trace) from None
        raise
    except AgentError:
        raise
    except LLMError as exc:
        raise AgentError(exc.status_code, exc.code, exc.message, trace) from exc
    except Exception as exc:  # a bug in a tool or the loop: keep the trace for debugging
        logger.exception("agent run failed")
        raise AgentError(500, "internal_error", "internal error in agent run", trace) from exc


async def _loop(
    llm: LLMClient,
    tools: ToolExecutor,
    request: ReformulateRequest,
    trace: Trace,
    max_iterations: int,
) -> AgentResult:
    messages = [
        Message("system", prompts.system_prompt()),
        Message("user", prompts.user_message(request)),
    ]
    evidence = _Evidence()
    last_call: tuple[str, str] | None = None
    final_only = False  # set by the loop guard: tools are no longer executed
    retried = False
    compact: dict[int, str] = {}  # index in messages -> shortened tool result for later calls

    for iteration in range(1, max_iterations + 1):
        sent = _history_for_call(messages, compact)
        tokens = approx_tokens(sent)  # what this call sends: where the provider limit bites
        response = await llm.complete_with_tools(sent, TOOL_SPECS)
        messages.append(response.message)

        if response.tool_calls:
            for call in response.tool_calls:
                signature = (call.name, json.dumps(call.arguments, sort_keys=True))
                if final_only or signature == last_call:
                    final_only = True
                    result = _skipped(call, trace, iteration, tokens)
                else:
                    result = await tools.execute(call.name, call.arguments)
                    evidence.add(call.name, call.arguments, result)
                    _record(trace, _trace_call(iteration, call, result), tokens)
                last_call = signature
                compact[len(messages)] = _to_json(compact_result(call.name, result))
                messages.append(
                    Message(
                        "tool",
                        _to_json(result),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            if final_only:
                messages.append(Message("user", prompts.FORCE_FINAL_MESSAGE))
            continue

        try:
            answer = _parse_final(response.text or "", evidence, request)
        except (ValueError, ValidationError) as exc:
            error = _describe(exc)
            _record(
                trace,
                {"type": "validation_error", "iteration": iteration, "error": error},
                tokens,
            )
            if retried:
                raise AgentError(
                    502, "agent_invalid_output", f"invalid final answer: {error}", trace
                ) from exc
            retried = True
            messages.append(Message("user", prompts.VALIDATION_RETRY_MESSAGE.format(error=error)))
            continue
        _record(trace, {"type": "final_answer", "iteration": iteration}, tokens)
        return AgentResult(answer=answer, trace=trace, iterations=iteration)

    raise AgentTimeoutError(f"no final answer after {max_iterations} iterations", trace)


@dataclass(frozen=True)
class _CalcCall:
    grams: list[float]
    per_100g: dict[str, float | None]
    without_data: list[str]  # ingredient names with grams > 0 and no nutrients


class _Evidence:
    """What tools actually returned in this run: citable sources and computed nutrition."""

    def __init__(self) -> None:
        self.sources: set[str] = set()
        # One entry per successful calc_nutrition call.
        self.nutrition: list[_CalcCall] = []
        self.product_allergens: dict[str, set[str]] = {}  # OFF source_id -> EU allergen codes

    def add(self, tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        if tool == "search_knowledge_base":
            self.sources |= {r["doc_id"] for r in result.get("results", [])}
        elif tool == "lookup_product" and result.get("found"):
            self.sources.add(result["source_id"])
            self.product_allergens[result["source_id"]] = set(result.get("allergens") or [])
        elif tool == "calc_nutrition" and "per_100g" in result:
            ingredients = arguments.get("ingredients", [])
            self.nutrition.append(
                _CalcCall(
                    grams=[float(i.get("grams", 0)) for i in ingredients],
                    per_100g=result["per_100g"],
                    # Ingredients sent with no nutrient at all: they count as 0 in the result.
                    without_data=[
                        str(i.get("name"))
                        for i in ingredients
                        if float(i.get("grams", 0)) > 0
                        and not any(
                            v is not None for v in (i.get("nutrients_per_100g") or {}).values()
                        )
                    ],
                )
            )


# EU allergens of animal origin: cannot remain in a vegan recipe.
ANIMAL_ALLERGENS = {"milk", "eggs", "fish", "crustaceans", "molluscs"}


def _parse_final(text: str, evidence: _Evidence, request: ReformulateRequest) -> ReformulationDraft:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # tolerate ```json fences
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg} at char {exc.pos})") from exc
    answer = ReformulationDraft.model_validate(data)

    # Grounding: the answer may only use what tools returned in this run.
    unknown = {s for sub in answer.substitutions for s in sub.sources} - evidence.sources
    if unknown:
        raise ValueError(
            f"sources {sorted(unknown)} were not returned by any tool call in this run"
        )
    _check_nutrition(answer, evidence, request)
    _check_allergens(answer, evidence, request)
    return answer


def _check_allergens(
    answer: ReformulationDraft, evidence: _Evidence, request: ReformulateRequest
) -> None:
    after = set(answer.allergens_after)
    # A product from Open Food Facts brings its allergens into the recipe.
    for sub in answer.substitutions:
        for source in sub.sources:
            missing = evidence.product_allergens.get(source, set()) - after
            if missing:
                raise ValueError(
                    f"{sub.replacement!r} cites {source}, which contains {sorted(missing)}: "
                    "add them to allergens_after (and to warnings if they are new)"
                )
    if request.goal == "remove_allergen" and request.goal_params["allergen"] in after:
        raise ValueError(
            f"goal is to remove {request.goal_params['allergen']!r} but it is still in "
            "allergens_after: replace every ingredient that contains it"
        )
    if request.goal == "make_vegan" and (animal := after & ANIMAL_ALLERGENS):
        raise ValueError(
            f"goal is a vegan recipe but allergens_after has {sorted(animal)} "
            "of animal origin: replace those ingredients"
        )


def _check_nutrition(
    answer: ReformulationDraft, evidence: _Evidence, request: ReformulateRequest
) -> None:
    """ "before" = calc_nutrition on the original recipe, "after" = a different calc call.

    The original recipe is recognised by its ingredient grams (the model may rename
    ingredients, e.g. translate them). Without this, one calc result copied into both
    fields would pass: it did in a live run.
    """
    original = _grams_key(i.grams for i in request.ingredients)
    before, after = answer.nutrition_per_100g.before, answer.nutrition_per_100g.after
    before_calls = {
        n
        for n, c in enumerate(evidence.nutrition)
        if _grams_key(c.grams) == original and _same_nutrition(before, c.per_100g)
    }
    if not before_calls:
        recipe = ", ".join(f"{g:g} g" for g in original)
        raise ValueError(
            "nutrition_per_100g.before must be the per_100g of a calc_nutrition call on the "
            f"original recipe (same ingredient grams: {recipe})"
        )
    # Missing data is checked before the "separate call" rule: it is the real problem
    # when two data-less calls both give zeros, and the message tells the model what to fix.
    _require_data("before", before_calls, evidence)
    after_any = {n for n, c in enumerate(evidence.nutrition) if _same_nutrition(after, c.per_100g)}
    if after_any:
        _require_data("after", after_any, evidence)
    if not after_any - before_calls:
        raise ValueError(
            "nutrition_per_100g.after must be the per_100g of a separate calc_nutrition call "
            "on the new recipe, not the call used for before"
        )


def _require_data(label: str, calls: set[int], evidence: _Evidence) -> None:
    """Zeros from ingredients sent without any nutrient are not nutrition data."""
    if all(evidence.nutrition[n].without_data for n in calls):
        names = ", ".join(evidence.nutrition[min(calls)].without_data)
        raise ValueError(
            f"nutrition_per_100g.{label} comes from calc_nutrition with no nutrients for: "
            f"{names}. Use nutrients_per_100g from search_knowledge_base results (or "
            "lookup_product), then call calc_nutrition again"
        )


def _grams_key(grams: Any) -> list[float]:
    return sorted(round(float(g), 1) for g in grams)


def _same_nutrition(values: NutritionValues, calc: dict[str, float | None]) -> bool:
    given = {n: v for n in NUTRIENTS if (v := getattr(values, n)) is not None}
    return bool(given) and all(
        calc.get(n) is not None and abs(v - calc[n]) <= 0.05 for n, v in given.items()
    )


def _to_json(data: Any) -> str:
    """Compact JSON for the model: no spaces after separators, non-ASCII kept as is."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _history_for_call(messages: list[Message], compact: dict[int, str]) -> list[Message]:
    """Messages to send: tool results from earlier rounds are replaced by their compact form.

    The latest round (tool results after the last assistant turn) goes in full, so the model
    reads each result completely once. The full results stay in `messages` and in the
    evidence used for validation.
    """
    last_assistant = max((i for i, m in enumerate(messages) if m.role == "assistant"), default=-1)
    return [
        replace(m, content=compact[i]) if i in compact and i < last_assistant else m
        for i, m in enumerate(messages)
    ]


def approx_tokens(messages: list[Message]) -> int:
    """Rough size of the message history in tokens: characters / 4.

    Counts message text and tool-call arguments; the tool schemas sent with every call
    are not included.
    """
    chars = 0
    for m in messages:
        chars += len(m.content)
        for c in m.tool_calls:
            chars += len(c.name) + len(json.dumps(c.arguments, ensure_ascii=False))
    return chars // 4


def _record(trace: Trace, entry: dict[str, Any], history_tokens: int) -> None:
    """Append to the trace and mirror it into the JSON logs (request_id is added there)."""
    entry["history_tokens"] = history_tokens
    trace.append(entry)
    keys = ("iteration", "tool", "call", "skipped", "error", "history_tokens")
    fields = {k: entry[k] for k in keys if k in entry}
    logger.info("agent %s", entry["type"], extra={"agent_" + k: v for k, v in fields.items()})


def _skipped(call: ToolCall, trace: Trace, iteration: int, tokens: int) -> dict[str, Any]:
    _record(trace, {**_trace_call(iteration, call, None), "skipped": "repeated_call"}, tokens)
    return {
        "error": {
            "code": "final_answer_required",
            "message": "Tool not executed: repeated call. Reply with the final JSON answer.",
        }
    }


def _trace_call(iteration: int, call: ToolCall, result: dict[str, Any] | None) -> dict[str, Any]:
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in call.arguments.items())
    entry = {
        "type": "tool_call",
        "iteration": iteration,
        "call": f"{call.name}({args})"[:300],
        "tool": call.name,
        "arguments": call.arguments,
    }
    if result is not None:
        entry["result"] = _summarize(call.name, result)
    return entry


def _summarize(tool: str, result: dict[str, Any]) -> Any:
    """Short, readable result for the trace (full results stay in the LLM history)."""
    if "error" in result:
        return {"error": result["error"]["code"]}
    if tool == "search_knowledge_base":
        return [f"{r['doc_id']} ({r['score']})" for r in result.get("results", [])]
    if tool == "lookup_product":
        if not result.get("found"):
            return "not found"
        return {k: result.get(k) for k in ("source_id", "product_name", "allergens")}
    if tool == "calc_nutrition":
        return {"per_100g": result.get("per_100g"), "warnings": result.get("warnings")}
    return json.dumps(result, ensure_ascii=False)[:200]


def _describe(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
            for e in exc.errors()
        )
    return str(exc)
