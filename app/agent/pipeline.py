"""Fixed pipeline (AGENT_MODE=pipeline): the code calls the tools, the LLM is called 2-3 times.

1. code:   search_knowledge_base for the goal, for every ingredient (its own spec) and for a
           replacement of every ingredient
2. LLM #1: choose substitutions, one ingredient per line, and say where each ingredient's
           nutrients come from
3. code:   lookup_product for ingredients without spec data, then check every number against
           its source; if data is still missing or the choice is invalid, LLM #2 re-chooses
4. code:   calc_nutrition for the original and the new recipe
5. LLM:    final answer in the loop's schema, checked by the same _parse_final;
           retried once if calls remain. Never more than MAX_LLM_CALLS calls.

The fallback from the spec for when free tool calling is unstable: the number of LLM calls is
bounded, the model never emits tool calls, and every nutrient number comes from a tool result.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.agent import prompts
from app.agent.loop import (
    AgentError,
    AgentResult,
    ToolExecutor,
    Trace,
    _describe,
    _Evidence,
    _parse_final,
    _record,
    _to_json,
    _trace_call,
    approx_tokens,
    guarded,
)
from app.agent.tools import NUTRIENTS
from app.llm.base import LLMClient, Message, ToolCall
from app.schemas import ReformulateRequest

MAX_LLM_CALLS = 3
MAX_CHOOSE_CALLS = 2  # a second choice when the first one is invalid or lacks data
KB_TEXT_FOR_CHOICE = 200  # characters of each document shown to the choosing LLM

Nutrients = dict[str, float | None]


class _Source(BaseModel):
    per_100g: Nutrients | None = None
    source: str | None = None
    off_query: str | None = None

    @field_validator("off_query")
    @classmethod
    def _english(cls, value: str | None) -> str | None:
        # Open Food Facts finds almost nothing by Ukrainian names ("заморожена полуниця").
        if value and not value.isascii():
            raise ValueError(f"off_query must be an English product name, got {value!r}")
        return value


class _Pick(BaseModel):
    original: str | None = None  # None: an added ingredient that replaces nothing
    replacement: str = Field(min_length=1)
    grams: float = Field(ge=0)
    sources: list[str] = Field(default_factory=list)

    @field_validator("original", "replacement")
    @classmethod
    def _one_ingredient(cls, value: str | None) -> str | None:
        if value and "+" in value:
            raise ValueError("one ingredient per substitution, not a combination with '+'")
        return value


class _Choice(BaseModel):
    substitutions: list[_Pick] = Field(min_length=1)
    nutrients: dict[str, _Source] = Field(default_factory=dict)


async def run_pipeline(
    llm: LLMClient, tools: ToolExecutor, request: ReformulateRequest, *, timeout_s: float
) -> AgentResult:
    trace: Trace = []
    return await guarded(_Pipeline(llm, tools, request, trace).run(), trace, timeout_s)


class _Pipeline:
    def __init__(
        self, llm: LLMClient, tools: ToolExecutor, request: ReformulateRequest, trace: Trace
    ) -> None:
        self.llm, self.tools, self.request, self.trace = llm, tools, request, trace
        self.evidence = _Evidence()
        self.pending: list[tuple[ToolCall, dict[str, Any]]] = []  # recorded with next LLM call
        self.llm_calls = 0
        self.docs: dict[str, dict[str, Any]] = {}  # doc_id -> search result item
        self.products: dict[str, dict[str, Any]] = {}  # OFF source_id -> lookup result
        self._resolved: dict[str, Nutrients] = {}  # name key -> verified per_100g
        # original ingredient key -> doc_id of the first spec with nutrients in its own search
        self.own_spec: dict[str, str] = {}

    async def run(self) -> AgentResult:
        await self._gather_evidence()

        choice, problems = await self._choose(feedback=None)
        if problems and self.llm_calls < MAX_CHOOSE_CALLS:
            choice, problems = await self._choose(feedback=problems)
        if problems or choice is None:
            raise AgentError(
                502, "agent_missing_data", "cannot build the recipes: " + "; ".join(problems),
                self.trace,
            )  # fmt: skip
        nutrients = self._resolved

        before = [
            {"name": i.name, "grams": i.grams, "nutrients_per_100g": nutrients[_key(i.name)]}
            for i in self.request.ingredients
        ]
        after = _apply(before, choice, nutrients)
        calc_before = await self._tool("calc_nutrition", {"ingredients": before})
        calc_after = await self._tool("calc_nutrition", {"ingredients": after})
        return await self._final(choice, calc_before, calc_after)

    # ---------- step 1: evidence, chosen by code ----------

    async def _gather_evidence(self) -> None:
        searches: list[tuple[str, str | None]] = [(prompts.pipeline_query(self.request), None)]
        for ingredient in self.request.ingredients:
            searches.append((ingredient.name, ingredient.name))  # its own spec (the original)
            searches.append(
                (prompts.pipeline_replacement_query(self.request, ingredient.name), None)
            )
        for query, own_of in searches:
            result = await self._tool("search_knowledge_base", {"query": query, "top_k": 3})
            for item in result.get("results", []):
                self.docs.setdefault(item["doc_id"], item)
                if own_of and _key(own_of) not in self.own_spec and item.get("nutrients_per_100g"):
                    self.own_spec[_key(own_of)] = item["doc_id"]

    # ---------- steps 2-3: choice, Open Food Facts, verified numbers ----------

    async def _choose(self, feedback: list[str] | None) -> tuple["_Choice | None", list[str]]:
        user = _choose_message(self.request, self.docs.values(), self.own_spec)
        if feedback:
            user += "\n\nYour previous choice was rejected:\n- " + "\n- ".join(feedback)
            with_data = [
                f"[{d['doc_id']}] {d['title']}"
                for d in self.docs.values()
                if d.get("nutrients_per_100g")
            ]
            user += (
                "\nFix it: pick replacements that have data in the evidence, or give off_query."
                "\nSpecs with nutrients: " + ", ".join(with_data)
            )
        messages = [Message("system", prompts.PIPELINE_CHOOSE_PROMPT), Message("user", user)]
        text = await self._ask(messages, "choose")
        try:
            choice = _Choice.model_validate(json.loads(_strip_fences(text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            error = _describe(exc) if isinstance(exc, ValidationError) else f"not JSON ({exc.msg})"
            return None, self._problems([f"invalid choice: {error}"], messages)

        self._record_choice(choice, messages)
        await self._fill_from_open_food_facts(choice)
        self._resolved, missing = self._verified_nutrients(choice)
        unknown = sorted({s for p in choice.substitutions for s in p.sources} - set(self.docs))
        problems = [f"no verified nutrients per 100 g for: {name}" for name in missing]
        if unknown:
            problems.append(f"sources {unknown} are not in the evidence")
        return choice, self._problems(problems, messages)

    def _record_choice(self, choice: _Choice, messages: list[Message]) -> None:
        """What the model chose, so a failed run can be read from the trace alone."""
        self._record(
            {
                "type": "choice",
                "substitutions": [
                    f"{p.original or '+'} -> {p.replacement} ({p.grams:g} g) {p.sources}"
                    for p in choice.substitutions
                ],
                "nutrient_sources": {
                    k: v.source or v.off_query for k, v in choice.nutrients.items()
                },
            },
            messages,
        )

    def _problems(self, problems: list[str], messages: list[Message]) -> list[str]:
        if problems:
            self._record({"type": "validation_error", "error": "; ".join(problems)}, messages)
        return problems

    async def _fill_from_open_food_facts(self, choice: _Choice) -> None:
        for info in choice.nutrients.values():
            if self._verify(info) is not None or not info.off_query:
                continue
            result = await self._tool("lookup_product", {"name": info.off_query})
            if result.get("found") and result.get("nutrients_per_100g"):
                self.products[result["source_id"]] = result
                info.source, info.per_100g = result["source_id"], None

    def _verified_nutrients(self, choice: _Choice) -> tuple[dict[str, Nutrients], list[str]]:
        """Numbers for every ingredient of both recipes, taken from their sources, never
        from the model's text. Returns (name key -> per_100g, names without data)."""
        by_key = {_key(k): v for k, v in choice.nutrients.items()}
        names = [i.name for i in self.request.ingredients]
        names += [p.replacement for p in choice.substitutions if p.grams > 0]
        resolved, missing = {}, []
        originals = {_key(i.name) for i in self.request.ingredients}
        for name in dict.fromkeys(names):
            verified = self._verify(by_key.get(_key(name)))
            if verified is None and _key(name) in originals and _key(name) in self.own_spec:
                # The original recipe is a fact, not the model's choice: when the model did not
                # map an original ingredient, use the spec its own search found.
                verified = self._own_spec_nutrients(name)
            if verified is None:
                missing.append(name)
            else:
                resolved[_key(name)] = verified
        return resolved, missing

    def _own_spec_nutrients(self, name: str) -> Nutrients | None:
        doc_id = self.own_spec[_key(name)]
        variants = parse_nutrient_variants(self.docs[doc_id].get("nutrients_per_100g") or "")
        if len(variants) > 1:
            # Specs list the standard product first (SPEC-009: the dairy working starter).
            # An assumption, so it is written to the trace.
            _record(
                self.trace,
                {
                    "type": "assumption",
                    "iteration": self.llm_calls,
                    "ingredient": name,
                    "note": f"nutrients from {doc_id}, variant 1 of {len(variants)}",
                },
                0,
            )
        return variants[0] if variants else None

    def _verify(self, info: _Source | None) -> Nutrients | None:
        if info is None or not info.source:
            return None  # numbers without a source are not accepted
        if info.source in self.products:
            return dict(self.products[info.source]["nutrients_per_100g"])
        text = self.docs.get(info.source, {}).get("nutrients_per_100g")
        variants = parse_nutrient_variants(text or "")
        claimed = {k: v for k, v in (info.per_100g or {}).items() if v is not None}
        for variant in variants:
            if claimed and all(
                k in variant and abs(v - variant[k]) <= 0.05 for k, v in claimed.items()
            ):
                return variant
        # One variant only: the source is unambiguous, so its numbers win over the model's.
        return variants[0] if len(variants) == 1 else None

    # ---------- step 5: final answer ----------

    async def _final(self, choice: _Choice, calc_before: dict, calc_after: dict) -> AgentResult:
        messages = [
            Message("system", prompts.pipeline_final_prompt()),
            Message(
                "user",
                _final_message(
                    self.request, choice, self.docs, self.products, calc_before, calc_after
                ),
            ),
        ]
        while True:
            text = await self._ask(messages, "final")
            try:
                answer = _parse_final(text, self.evidence, self.request)
            except (ValueError, ValidationError) as exc:
                error = _describe(exc)
                self._record({"type": "validation_error", "error": error}, messages)
                if self.llm_calls >= MAX_LLM_CALLS:
                    raise AgentError(
                        502, "agent_invalid_output", f"invalid final answer: {error}", self.trace
                    ) from exc
                messages += [
                    Message("assistant", text),
                    Message("user", prompts.VALIDATION_RETRY_MESSAGE.format(error=error)),
                ]
                continue
            self._record({"type": "final_answer"}, messages)
            return AgentResult(answer=answer, trace=self.trace, iterations=self.llm_calls)

    # ---------- plumbing ----------

    async def _tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.tools.execute(name, arguments)
        self.evidence.add(name, arguments, result)
        call_id = f"pipeline-{len(self.pending) + len(self.trace)}"
        self.pending.append((ToolCall(id=call_id, name=name, arguments=arguments), result))
        return result

    async def _ask(self, messages: list[Message], purpose: str) -> str:
        """One LLM call; tool calls made since the previous one are recorded under it."""
        self.llm_calls += 1
        tokens = approx_tokens(messages)
        for call, result in self.pending:
            _record(self.trace, _trace_call(self.llm_calls, call, result), tokens)
        self.pending.clear()
        entry = {"type": "llm_call", "iteration": self.llm_calls, "purpose": purpose}
        _record(self.trace, entry, tokens)
        return await self.llm.complete(messages, json_output=True)

    def _record(self, entry: dict[str, Any], messages: list[Message]) -> None:
        # Tool calls made after the last LLM call are logged before the entry that follows them.
        for call, result in self.pending:
            _record(self.trace, _trace_call(self.llm_calls, call, result), approx_tokens(messages))
        self.pending.clear()
        _record(self.trace, {**entry, "iteration": self.llm_calls}, approx_tokens(messages))


_NUMBER = re.compile(r"\b(kcal|protein_g|fat_g|carbs_g|sugar_g)\b\s*:?\s*(-?\d+(?:[.,]\d+)?)")


def parse_nutrient_variants(text: str) -> list[dict[str, float]]:
    """Nutrient sets in a spec's "Нутрієнти на 100 г" line, in order of appearance.

    One set per product variant: a set ends when a nutrient repeats ("Варіант А: kcal 58 …
    Варіант Б: kcal 375 …"). Sets without all five nutrients are dropped.
    """
    variants: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for name, value in _NUMBER.findall(text):
        if name in current:
            variants.append(current)
            current = {}
        current[name] = float(value.replace(",", "."))
    variants.append(current)
    return [v for v in variants if set(v) == set(NUTRIENTS)]


def _choose_message(request: ReformulateRequest, docs: Any, own_spec: dict[str, str]) -> str:
    matches = [
        f"- {i.name} -> [{own_spec[_key(i.name)]}]"
        if _key(i.name) in own_spec
        else f"- {i.name} -> no spec"
        for i in request.ingredients
    ]
    lines = []
    for d in docs:
        n = d.get("nutrients_per_100g")
        nutrients = f" | nutrients_per_100g: {n}" if n else ""
        lines.append(f"[{d['doc_id']}] {d['title']}{nutrients}\n{d['text'][:KB_TEXT_FOR_CHOICE]}")
    return (
        prompts.user_message(request)
        + "\n\nSpec found for each original ingredient (use its nutrients_per_100g):\n"
        + "\n".join(matches)
        + "\n\nEvidence:\n"
        + "\n\n".join(lines)
    )


def _final_message(
    request: ReformulateRequest,
    choice: _Choice,
    docs: dict[str, dict[str, Any]],
    products: dict[str, dict[str, Any]],
    calc_before: dict[str, Any],
    calc_after: dict[str, Any],
) -> str:
    evidence = [f"[{d['doc_id']}] {d['title']}" for d in docs.values()]
    evidence += [
        f"[{sid}] Open Food Facts: {p.get('product_name')}, allergens: {p.get('allergens')}"
        for sid, p in products.items()
    ]
    return (
        prompts.user_message(request)
        + "\n\nChosen substitutions (original null = added ingredient):\n"
        + _to_json([p.model_dump() for p in choice.substitutions])
        + "\n\nEvidence you may cite:\n"
        + "\n".join(evidence)
        + "\n\ncalc_nutrition, original recipe: "
        + _to_json({k: calc_before.get(k) for k in ("per_100g", "warnings")})
        + "\ncalc_nutrition, new recipe: "
        + _to_json({k: calc_after.get(k) for k in ("per_100g", "warnings")})
    )


def _key(name: str) -> str:
    return " ".join(name.lower().split())


def _apply(
    before: list[dict[str, Any]], choice: _Choice, nutrients: dict[str, Nutrients]
) -> list[dict[str, Any]]:
    """The new recipe: a substitution replaces its original ingredient, or is added."""
    after = [dict(i) for i in before]
    for pick in choice.substitutions:
        line = {
            "name": pick.replacement,
            "grams": pick.grams,
            "nutrients_per_100g": nutrients.get(_key(pick.replacement), {}),
        }
        index = next(
            (
                n
                for n, i in enumerate(after)
                if pick.original and _key(i["name"]) == _key(pick.original)
            ),
            None,
        )
        if index is None:
            after.append(line)
        else:
            after[index] = line
    return after


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
