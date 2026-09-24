"""Reformulation agent: a plain tool-calling loop, no frameworks.

One iteration = one LLM call. The loop stops with a validated answer, or raises
AgentError carrying the trace collected so far (agent_timeout on limits).
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from app.agent import prompts
from app.agent.tools import NUTRIENTS, TOOL_SPECS
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
    try:
        async with asyncio.timeout(timeout_s) as deadline:
            return await _loop(llm, tools, request, trace, max_iterations)
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

    for iteration in range(1, max_iterations + 1):
        response = await llm.complete_with_tools(messages, TOOL_SPECS)
        messages.append(response.message)

        if response.tool_calls:
            for call in response.tool_calls:
                signature = (call.name, json.dumps(call.arguments, sort_keys=True))
                if final_only or signature == last_call:
                    final_only = True
                    result = _skipped(call, trace, iteration)
                else:
                    result = await tools.execute(call.name, call.arguments)
                    evidence.add(call.name, result)
                    _record(trace, _trace_call(iteration, call, result))
                last_call = signature
                messages.append(
                    Message(
                        "tool",
                        json.dumps(result, ensure_ascii=False),
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
            _record(trace, {"type": "validation_error", "iteration": iteration, "error": error})
            if retried:
                raise AgentError(
                    502, "agent_invalid_output", f"invalid final answer: {error}", trace
                ) from exc
            retried = True
            messages.append(Message("user", prompts.VALIDATION_RETRY_MESSAGE.format(error=error)))
            continue
        _record(trace, {"type": "final_answer", "iteration": iteration})
        return AgentResult(answer=answer, trace=trace, iterations=iteration)

    raise AgentTimeoutError(f"no final answer after {max_iterations} iterations", trace)


class _Evidence:
    """What tools actually returned in this run: citable sources and computed nutrition."""

    def __init__(self) -> None:
        self.sources: set[str] = set()
        self.nutrition: list[dict[str, float | None]] = []
        self.product_allergens: dict[str, set[str]] = {}  # OFF source_id -> EU allergen codes

    def add(self, tool: str, result: dict[str, Any]) -> None:
        if tool == "search_knowledge_base":
            self.sources |= {r["doc_id"] for r in result.get("results", [])}
        elif tool == "lookup_product" and result.get("found"):
            self.sources.add(result["source_id"])
            self.product_allergens[result["source_id"]] = set(result.get("allergens") or [])
        elif tool == "calc_nutrition" and "per_100g" in result:
            self.nutrition.append(result["per_100g"])


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
    for label in ("before", "after"):
        values = getattr(answer.nutrition_per_100g, label)
        if not any(_same_nutrition(values, calc) for calc in evidence.nutrition):
            raise ValueError(
                f"nutrition_per_100g.{label} does not match any calc_nutrition result: "
                "call calc_nutrition and copy its per_100g values"
            )
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


def _same_nutrition(values: NutritionValues, calc: dict[str, float | None]) -> bool:
    given = {n: v for n in NUTRIENTS if (v := getattr(values, n)) is not None}
    return bool(given) and all(
        calc.get(n) is not None and abs(v - calc[n]) <= 0.05 for n, v in given.items()
    )


def _record(trace: Trace, entry: dict[str, Any]) -> None:
    """Append to the trace and mirror it into the JSON logs (request_id is added there)."""
    trace.append(entry)
    fields = {k: entry[k] for k in ("iteration", "tool", "call", "skipped", "error") if k in entry}
    logger.info("agent %s", entry["type"], extra={"agent_" + k: v for k, v in fields.items()})


def _skipped(call: ToolCall, trace: Trace, iteration: int) -> dict[str, Any]:
    _record(trace, {**_trace_call(iteration, call, None), "skipped": "repeated_call"})
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
