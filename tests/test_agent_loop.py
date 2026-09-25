"""Agent loop with FakeLLM and fake tools: no network, no database, no keys."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from app.agent import prompts
from app.agent.loop import AgentError, AgentTimeoutError, approx_tokens, run_agent
from app.agent.tools import CalcNutritionArgs, calc_nutrition
from app.llm.base import LLMError, ToolCall
from app.llm.fake import FakeLLM
from app.schemas import ReformulateRequest, ReformulationDraft

REQUEST = ReformulateRequest(
    product_name="Полуничний йогурт 2.5%",
    ingredients=[
        {"name": "молоко 2.5%", "grams": 800},
        {"name": "цукор", "grams": 90},
        {"name": "полуниця заморожена", "grams": 100},
        {"name": "закваска", "grams": 10},
    ],
    goal="remove_allergen",
    goal_params={"allergen": "milk"},
)


def _ing(name, grams, kcal, protein, fat, carbs, sugar):
    return {
        "name": name,
        "grams": grams,
        "nutrients_per_100g": {
            "kcal": kcal,
            "protein_g": protein,
            "fat_g": fat,
            "carbs_g": carbs,
            "sugar_g": sugar,
        },
    }


BEFORE = [
    _ing("молоко 2.5%", 800, 52, 2.8, 2.5, 4.7, 4.7),
    _ing("цукор", 90, 400, 0, 0, 100, 100),
    _ing("полуниця", 100, 32, 0.7, 0.3, 7.7, 4.9),
    _ing("закваска", 10, 58, 3.1, 2.4, 5.2, 3.9),
]
AFTER = [
    _ing("кокосове молоко 2.5%", 790, 28, 0.2, 2.5, 1.2, 1.0),
    _ing("тапіоковий крохмаль", 20, 350, 0, 0, 87, 0),
    _ing("цукор", 90, 400, 0, 0, 100, 100),
    _ing("полуниця", 100, 32, 0.7, 0.3, 7.7, 4.9),
]
PER_100G_BEFORE = calc_nutrition(CalcNutritionArgs(ingredients=BEFORE).ingredients)["per_100g"]
PER_100G_AFTER = calc_nutrition(CalcNutritionArgs(ingredients=AFTER).ingredients)["per_100g"]


class FakeTools:
    """Canned search/lookup, real calc_nutrition; optional delay for chosen tools."""

    def __init__(self, slow: dict[str, float] | None = None) -> None:
        self.executed: list[tuple[str, dict]] = []
        self.slow = slow or {}

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        await asyncio.sleep(self.slow.get(name, 0))
        if name == "search_knowledge_base":
            return {
                "results": [
                    {"doc_id": "SPEC-002", "title": "Кокосове молоко", "text": "…", "score": 0.88},
                    {
                        "doc_id": "TRIAL-001",
                        "title": "Йогурт на кокосі",
                        "text": "…",
                        "score": 0.87,
                    },
                ]
            }
        if name == "lookup_product":
            return {
                "found": True,
                "source_id": "OFF:111",
                "product_name": "Soy Drink",
                "allergens": ["soybeans"],
            }
        if name == "calc_nutrition":
            return calc_nutrition(CalcNutritionArgs.model_validate(arguments).ingredients)
        return {"error": {"code": "unknown_tool", "message": name}}


def call(tool: str, /, **arguments) -> list[ToolCall]:
    return [ToolCall(id=f"{tool}-{len(json.dumps(arguments))}", name=tool, arguments=arguments)]


def final(sources=("SPEC-002", "TRIAL-001"), before=None, **overrides) -> str:
    answer = {
        "substitutions": [
            {
                "original": "молоко 2.5%",
                "replacement": "кокосове молоко 2.5% жиру",
                "grams": 790,
                "rationale": "Схожа жирність, перевірено в TRIAL-001.",
                "sources": list(sources),
            }
        ],
        "allergens_before": ["milk"],
        "allergens_after": [],
        "nutrition_per_100g": {"before": before or PER_100G_BEFORE, "after": PER_100G_AFTER},
        "warnings": ["Білок падає приблизно на 90%."],
    }
    return json.dumps({**answer, **overrides}, ensure_ascii=False)


# The usual successful path: search, two calc_nutrition calls, final answer.
HAPPY = [
    call("search_knowledge_base", query="рослинна заміна молока у ферментованому продукті"),
    call("calc_nutrition", ingredients=BEFORE),
    call("calc_nutrition", ingredients=AFTER),
]


async def run(replies, tools=None, max_iterations=6, timeout_s=5.0):
    llm = FakeLLM(replies)
    tools = tools or FakeTools()
    result = await run_agent(
        llm, tools, REQUEST, max_iterations=max_iterations, timeout_s=timeout_s, prepare_facts=False
    )
    return result, llm, tools


async def test_successful_run_calls_tools_and_records_trace():
    result, llm, tools = await run([*HAPPY, final()])

    assert result.iterations == 4
    assert [name for name, _ in tools.executed] == [
        "search_knowledge_base",
        "calc_nutrition",
        "calc_nutrition",
    ]
    assert result.answer.nutrition_per_100g.before.kcal == 81.4
    assert result.answer.substitutions[0].sources == ["SPEC-002", "TRIAL-001"]

    assert [e["type"] for e in result.trace] == ["tool_call"] * 3 + ["final_answer"]
    first = result.trace[0]
    assert first["call"].startswith("search_knowledge_base(query=")
    assert first["result"] == ["SPEC-002 (0.88)", "TRIAL-001 (0.87)"]
    assert result.trace[1]["result"]["per_100g"] == PER_100G_BEFORE

    # The tool result was fed back to the model as a tool message.
    fed_back = llm.calls[1][-1]
    assert fed_back.role == "tool"
    assert "SPEC-002" in fed_back.content
    assert fed_back.tool_call_id == HAPPY[0][0].id


async def test_stops_at_iteration_limit_with_trace():
    endless = [call("search_knowledge_base", query=f"варіант {i}") for i in range(10)]

    with pytest.raises(AgentTimeoutError) as err:
        await run(endless, max_iterations=6)

    assert err.value.status_code == 504
    assert err.value.code == "agent_timeout"
    assert "6 iterations" in err.value.message
    assert len(err.value.trace) == 6
    assert err.value.trace[-1]["arguments"] == {"query": "варіант 5"}


async def test_wall_clock_timeout_keeps_partial_trace():
    tools = FakeTools(slow={"calc_nutrition": 1.0})

    with pytest.raises(AgentTimeoutError) as err:
        await run([*HAPPY, final()], tools=tools, timeout_s=0.2)

    assert "0.2 s" in err.value.message
    assert [e["tool"] for e in err.value.trace] == ["search_knowledge_base"]


async def test_invalid_json_is_retried_once_with_the_error():
    result, llm, _ = await run([*HAPPY, "Ось моя відповідь: {не json", final()])

    assert result.iterations == 5
    errors = [e for e in result.trace if e["type"] == "validation_error"]
    assert len(errors) == 1
    assert "not valid JSON" in errors[0]["error"]
    retry_prompt = llm.calls[4][-1]
    assert retry_prompt.role == "user"
    assert "rejected" in retry_prompt.content
    assert "not valid JSON" in retry_prompt.content


async def test_second_invalid_answer_fails_with_trace():
    with pytest.raises(AgentError) as err:
        await run([*HAPPY, "not json", '{"substitutions": []}'])

    assert err.value.status_code == 502
    assert err.value.code == "agent_invalid_output"
    assert [e["type"] for e in err.value.trace].count("validation_error") == 2


async def test_repeated_call_is_not_executed_and_forces_final_answer():
    same = call("search_knowledge_base", query="аквафаба")
    result, llm, tools = await run(
        [
            call("calc_nutrition", ingredients=BEFORE),
            call("calc_nutrition", ingredients=AFTER),
            same,
            same,
            final(),
        ]
    )

    assert [name for name, _ in tools.executed].count("search_knowledge_base") == 1
    skipped = [e for e in result.trace if e.get("skipped")]
    assert len(skipped) == 1
    assert skipped[0]["iteration"] == 4
    assert llm.calls[4][-1].content == prompts.FORCE_FINAL_MESSAGE


async def test_tool_calls_after_forced_final_are_ignored():
    same = call("search_knowledge_base", query="аквафаба")
    result, _, tools = await run(
        [*HAPPY[1:], same, same, call("lookup_product", name="coconut milk"), final(("SPEC-002",))]
    )

    assert "lookup_product" not in [name for name, _ in tools.executed]
    assert [e["tool"] for e in result.trace if e.get("skipped")] == [
        "search_knowledge_base",
        "lookup_product",
    ]


async def test_made_up_source_is_rejected_then_fixed():
    result, _, _ = await run([*HAPPY, final(sources=["SPEC-014"]), final()])

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "SPEC-014" in error["error"]
    assert result.answer.substitutions[0].sources == ["SPEC-002", "TRIAL-001"]


async def test_nutrition_not_from_calc_nutrition_is_rejected():
    guessed = {"kcal": 78, "protein_g": 2.9, "sugar_g": 12.1}
    result, _, _ = await run([*HAPPY, final(before=guessed), final()])

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "nutrition_per_100g.before" in error["error"]
    assert "calc_nutrition" in error["error"]


async def test_llm_error_keeps_trace():
    def fail(_messages):
        raise LLMError("gemini rate limit", "llm_rate_limited", 503)

    with pytest.raises(AgentError) as err:
        await run([HAPPY[0], fail])

    assert err.value.code == "llm_rate_limited"
    assert err.value.status_code == 503
    assert [e["tool"] for e in err.value.trace] == ["search_knowledge_base"]


def test_substitution_without_sources_requires_low_confidence():
    data = json.loads(final(sources=[]))
    with pytest.raises(ValidationError, match="no sources"):
        ReformulationDraft.model_validate(data)

    data["substitutions"][0]["confidence"] = "low"
    draft = ReformulationDraft.model_validate(data)
    assert any("низьку впевненість" in w for w in draft.warnings)


@pytest.mark.parametrize(
    ("goal", "params"),
    [
        ("remove_allergen", {"allergen": "wheat"}),  # not an EU allergen code
        ("remove_allergen", {}),
        ("reduce_sugar", {"percent": 60}),
        ("make_vegan", {"strict": True}),
    ],
)
def test_goal_params_are_validated(goal, params):
    with pytest.raises(ValidationError):
        ReformulateRequest(
            product_name="x", ingredients=[{"name": "a", "grams": 1}], goal=goal, goal_params=params
        )


async def test_allergens_of_cited_off_product_must_appear_after():
    lookup = call("lookup_product", name="soy drink")
    cites_soy = final(sources=["SPEC-002", "OFF:111"])
    fixed = final(sources=["SPEC-002", "OFF:111"], allergens_after=["soybeans"])
    result, _, _ = await run([*HAPPY, lookup, cites_soy, fixed])

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "OFF:111" in error["error"]
    assert "soybeans" in error["error"]
    assert result.answer.allergens_after == ["soybeans"]


async def test_goal_allergen_must_be_gone():
    with pytest.raises(AgentError) as err:
        await run([*HAPPY, final(allergens_after=["milk"]), final(allergens_after=["milk"])])

    errors = [e["error"] for e in err.value.trace if e["type"] == "validation_error"]
    assert len(errors) == 2
    assert "remove 'milk'" in errors[0]


async def test_vegan_answer_cannot_keep_animal_allergens():
    vegan = REQUEST.model_copy(update={"goal": "make_vegan", "goal_params": {}})
    llm = FakeLLM([*HAPPY, final(allergens_after=["eggs"]), final()])
    result = await run_agent(llm, FakeTools(), vegan, max_iterations=6, timeout_s=5)

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "vegan" in error["error"]
    assert "eggs" in error["error"]


def test_compact_answer_format_names_every_field_of_the_answer_model():
    # The prompt no longer embeds the generated JSON schema; guard against drift.
    from typing import get_args

    from app.schemas import EUAllergen, NutritionValues, Substitution

    fields = {*ReformulationDraft.model_fields, *Substitution.model_fields}
    fields |= set(NutritionValues.model_fields) | set(get_args(EUAllergen))
    missing = {f for f in fields if f not in prompts.ANSWER_FORMAT}
    assert not missing
    assert len(prompts.ANSWER_FORMAT) // 4 <= 300  # approx tokens


async def test_every_trace_entry_has_history_size():
    result, llm, _ = await run([*HAPPY, "not json", final()])

    assert all(isinstance(e["history_tokens"], int) for e in result.trace)
    assert result.trace[0]["history_tokens"] == approx_tokens(llm.calls[0])


async def test_older_tool_results_are_sent_compact_latest_in_full():
    result, llm, _ = await run([*HAPPY, final()])

    # Call 2 sees the search result in full: it is the latest round.
    latest = llm.calls[1][-1]
    assert latest.role == "tool"
    assert '"text"' in latest.content
    # Calls 3 and 4 get the same search result without chunk text, doc_ids still citable.
    for later in llm.calls[2:]:
        search = next(m for m in later if m.role == "tool" and m.name == "search_knowledge_base")
        assert '"text"' not in search.content
        assert "SPEC-002" in search.content
    # Validation still used the full results: the answer citing SPEC-002 was accepted.
    assert result.answer.substitutions[0].sources == ["SPEC-002", "TRIAL-001"]


async def test_one_calc_result_copied_into_before_and_after_is_rejected():
    # Live bug: the model called calc_nutrition once and used the result for both fields.
    only_after = [call("calc_nutrition", ingredients=AFTER)]
    copied = final(before=PER_100G_AFTER)
    fixed_script = [call("calc_nutrition", ingredients=BEFORE), final()]

    result, _, _ = await run([HAPPY[0], *only_after, copied, *fixed_script])

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "original recipe" in error["error"]
    assert "800 g" in error["error"]
    assert result.answer.nutrition_per_100g.before.kcal == 81.4


async def test_after_must_come_from_a_different_calc_call():
    same_twice = final(before=PER_100G_BEFORE)
    same_twice = json.loads(same_twice)
    same_twice["nutrition_per_100g"]["after"] = PER_100G_BEFORE
    with pytest.raises(AgentError) as err:
        await run(
            [HAPPY[0], HAPPY[1], json.dumps(same_twice), json.dumps(same_twice)],
        )
    errors = [e["error"] for e in err.value.trace if e["type"] == "validation_error"]
    assert "separate calc_nutrition call" in errors[0]


async def test_before_calc_is_recognised_by_grams_even_if_names_are_translated():
    english = [{**i, "name": f"ingredient {n}"} for n, i in enumerate(BEFORE)]
    result, _, _ = await run(
        [HAPPY[0], call("calc_nutrition", ingredients=english), HAPPY[2], final()]
    )
    assert result.answer.nutrition_per_100g.before.kcal == 81.4


async def test_calc_with_ingredients_without_nutrients_is_rejected():
    # Live bug: calc_nutrition with {} for every ingredient gave zeros that passed.
    empty_before = [{**i, "nutrients_per_100g": {}} for i in BEFORE]
    zeros = calc_nutrition(CalcNutritionArgs(ingredients=empty_before).ingredients)["per_100g"]
    script = [
        HAPPY[0],
        call("calc_nutrition", ingredients=empty_before),
        HAPPY[2],
        final(before=zeros),
        HAPPY[1],
        final(),
    ]
    result, _, _ = await run(script, max_iterations=8)

    error = next(e for e in result.trace if e["type"] == "validation_error")
    assert "no nutrients for" in error["error"]
    assert "молоко 2.5%" in error["error"]
    assert result.answer.nutrition_per_100g.before.kcal == 81.4


async def test_partially_missing_nutrients_are_accepted():
    partial = [dict(BEFORE[0], nutrients_per_100g={"kcal": 52}), *BEFORE[1:]]
    per_100g = calc_nutrition(CalcNutritionArgs(ingredients=partial).ingredients)["per_100g"]
    result, _, _ = await run(
        [HAPPY[0], call("calc_nutrition", ingredients=partial), HAPPY[2], final(before=per_100g)]
    )
    assert result.answer.nutrition_per_100g.before.model_dump() == per_100g


async def test_two_data_less_calcs_get_the_missing_data_message_not_the_separation_one():
    # Live bug: both calls had {} nutrients, both gave zeros, and the model was told
    # "after must be a separate call" instead of what was actually wrong.
    empty_before = [{**i, "nutrients_per_100g": {}} for i in BEFORE]
    empty_after = [{**i, "nutrients_per_100g": {}} for i in AFTER]
    zeros = calc_nutrition(CalcNutritionArgs(ingredients=empty_before).ingredients)["per_100g"]
    script = [
        HAPPY[0],
        call("calc_nutrition", ingredients=empty_before),
        call("calc_nutrition", ingredients=empty_after),
        final(nutrition_per_100g={"before": zeros, "after": zeros}),
    ]
    with pytest.raises(AgentError) as err:
        await run([*script, script[-1]])

    first_error = next(e["error"] for e in err.value.trace if e["type"] == "validation_error")
    assert "no nutrients for" in first_error
    assert "nutrients_per_100g from search_knowledge_base" in first_error


def test_system_prompt_has_no_todo_and_contains_the_example():
    prompt = prompts.system_prompt()

    assert "TODO" not in prompt
    assert '"allergens_after": ["soybeans"]' in prompt
    assert "рослинна закваска DVS" in prompt  # the example replaces the hidden allergen too
    assert "{" not in prompts.GOAL_DESCRIPTIONS["make_vegan"]


def test_example_answer_is_a_valid_answer():
    # The example must itself pass the answer model, or it would teach a broken format.
    draft = ReformulationDraft.model_validate(json.loads(prompts.EXAMPLE_ANSWER))
    assert [s.original for s in draft.substitutions] == ["молоко 2.5%", "закваска"]


def test_goal_descriptions_are_rendered_into_the_user_message():
    message = prompts.user_message(REQUEST)
    assert "'milk'" in message
    assert "starter cultures" in message


async def test_parallel_tool_calls_in_one_turn_use_one_iteration():
    both_calcs = [*HAPPY[1], *HAPPY[2]]  # two calc_nutrition calls in the same turn
    result, llm, tools = await run([HAPPY[0], both_calcs, final()])

    assert result.iterations == 3
    assert [name for name, _ in tools.executed].count("calc_nutrition") == 2
    assert [e["iteration"] for e in result.trace if e.get("tool") == "calc_nutrition"] == [2, 2]
    # Both results went back to the model, one tool message per call id.
    fed_back = [m for m in llm.calls[2] if m.role == "tool" and m.name == "calc_nutrition"]
    assert [m.tool_call_id for m in fed_back] == [c.id for c in both_calcs]


def test_prompt_asks_for_parallel_tool_calls():
    assert "SAME turn as parallel calls" in prompts.system_prompt()


def test_added_ingredient_has_null_original():
    data = json.loads(final())
    data["substitutions"].append(
        {"original": None, "replacement": "ER-ST", "grams": 27, "rationale": "SPEC-007",
         "sources": ["SPEC-002"]}
    )  # fmt: skip
    draft = ReformulationDraft.model_validate(data)
    assert draft.substitutions[-1].original is None


# ---------- recipe facts and turn budget (what makes the loop fit in 6 turns) ----------


def _spec(doc_id: str, title: str, nutrients: str) -> dict:
    return {"doc_id": doc_id, "title": title, "text": "…", "score": 0.9,
            "nutrients_per_100g": nutrients}  # fmt: skip


ORIGINAL_SPECS = {
    "молоко 2.5%": _spec(
        "SPEC-001",
        "Молоко коров'яче 2.5% жиру",
        "kcal: 52; protein_g: 2.8; fat_g: 2.5; carbs_g: 4.7; sugar_g: 4.7",
    ),
    "цукор": _spec(
        "SPEC-006",
        "Цукор білий кристалічний",
        "kcal: 400; protein_g: 0; fat_g: 0; carbs_g: 100; sugar_g: 100",
    ),
}


class FactsTools(FakeTools):
    """Search that knows the specs of the original ingredients."""

    async def execute(self, name, arguments):
        if name == "search_knowledge_base" and arguments["query"] in ORIGINAL_SPECS:
            self.executed.append((name, arguments))
            return {"results": [ORIGINAL_SPECS[arguments["query"]]]}
        return await super().execute(name, arguments)


async def test_original_ingredient_facts_are_prepared_before_the_first_call():
    llm, tools = FakeLLM([*HAPPY, final()]), FactsTools()
    result = await run_agent(llm, tools, REQUEST, max_iterations=6, timeout_s=5)

    first_user = llm.calls[0][1].content
    assert prompts.RECIPE_FACTS_HEADER in first_user
    assert (
        "- молоко 2.5% (800 g): [SPEC-001] Молоко коров'яче 2.5% жиру | "
        "nutrients_per_100g: kcal: 52" in first_user
    )
    assert (
        "- полуниця заморожена (100 g): no spec, use lookup_product with an English name"
        in first_user
    )
    facts = [e for e in result.trace if e["iteration"] == 0]
    assert [e["tool"] for e in facts] == ["search_knowledge_base"] * 4  # one per ingredient
    assert result.iterations == 4  # facts cost no LLM calls


async def test_turn_budget_hints_before_the_last_two_turns():
    endless = [call("search_knowledge_base", query=f"варіант {i}") for i in range(10)]
    llm = FakeLLM(endless)
    with pytest.raises(AgentTimeoutError):
        await run_agent(
            llm, FakeTools(), REQUEST, max_iterations=6, timeout_s=5, prepare_facts=False
        )

    assert llm.calls[4][-1].content.startswith("Turn 5 of 6.")
    assert "calc_nutrition in THIS turn" in llm.calls[4][-1].content
    assert llm.calls[5][-1].content == "Turn 6 of 6, the last: reply with the final JSON only."
    assert all(m.role != "user" or "Turn" not in m.content for m in llm.calls[3][2:])


def test_gemini_merges_a_note_after_tool_results_into_the_same_user_turn():
    from app.llm.base import Message as M
    from app.llm.gemini import _to_contents

    contents = _to_contents([
        M("user", "recipe"),
        M("assistant", tool_calls=[ToolCall("a", "search_knowledge_base", {"query": "x"})]),
        M("tool", "{}", tool_call_id="a", name="search_knowledge_base"),
        M("user", "Turn 5 of 6."),
    ])  # fmt: skip
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[-1].parts[0].function_response is not None
    assert contents[-1].parts[1].text == "Turn 5 of 6."


class BarrierTools(FactsTools):
    """The first `parties` calls wait until all of them are in flight at once: sequential
    execution would time out here. Earlier calls then finish later, to check the order."""

    def __init__(self, parties: int) -> None:
        super().__init__()
        self.barrier = asyncio.Barrier(parties)
        self.started = 0

    async def execute(self, name, arguments):
        n, self.started = self.started, self.started + 1
        if n < self.barrier.parties:
            async with asyncio.timeout(1):
                await self.barrier.wait()
            await asyncio.sleep(0.01 * (self.barrier.parties - n))  # first call finishes last
        return await super().execute(name, arguments)


async def test_tool_calls_of_one_turn_run_concurrently_and_keep_call_order():
    turn = [
        *call("search_knowledge_base", query="рослинна заміна молока"),
        *call("lookup_product", name="coconut milk"),
        *call("calc_nutrition", ingredients=BEFORE),
    ]
    result, llm, _ = await run([turn, HAPPY[2], final()], tools=BarrierTools(parties=3))

    first = [e for e in result.trace if e["iteration"] == 1]
    assert [e["tool"] for e in first] == [c.name for c in turn]
    assert first[1]["result"]["source_id"] == "OFF:111"  # each result stays with its own call
    fed_back = [m for m in llm.calls[1] if m.role == "tool"]
    assert [m.tool_call_id for m in fed_back] == [c.id for c in turn]
    assert json.loads(fed_back[2].content)["per_100g"] == PER_100G_BEFORE


async def test_repeated_call_in_the_same_turn_skips_it_and_the_rest():
    same = call("search_knowledge_base", query="аквафаба")
    turn = [*same, *same, *call("lookup_product", name="aquafaba")]
    result, _, tools = await run([*HAPPY[1:], turn, final(("SPEC-002",))])

    assert [name for name, _ in tools.executed].count("search_knowledge_base") == 1
    assert "lookup_product" not in [name for name, _ in tools.executed]
    assert [e["tool"] for e in result.trace if e.get("skipped")] == [
        "search_knowledge_base",
        "lookup_product",
    ]


async def test_recipe_facts_are_searched_concurrently_in_ingredient_order():
    llm = FakeLLM([*HAPPY, final()])
    result = await run_agent(llm, BarrierTools(parties=4), REQUEST, max_iterations=6, timeout_s=5)

    facts = [e for e in result.trace if e["iteration"] == 0]
    assert [e["arguments"]["query"] for e in facts] == [i.name for i in REQUEST.ingredients]
    lines = llm.calls[0][1].content.split(prompts.RECIPE_FACTS_HEADER)[1].strip().splitlines()
    assert [line.split(" (")[0] for line in lines] == [f"- {i.name}" for i in REQUEST.ingredients]


# ---------- structured spec data (KAN-16) ----------


class MismatchTools(FakeTools):
    """calc_nutrition flags the original milk numbers as not matching the spec in the DB."""

    async def execute(self, name, arguments):
        result = await super().execute(name, arguments)
        if name == "calc_nutrition" and arguments["ingredients"][0]["name"] == "молоко 2.5%":
            result["data_mismatches"] = [
                {"ingredient": "молоко 2.5%", "doc_id": "SPEC-001",
                 "given": {"protein_g": 3.5}, "expected": {"protein_g": 2.8}}
            ]  # fmt: skip
        return result


async def test_faked_input_nutrients_end_up_in_warnings():
    result, _, _ = await run([*HAPPY, final()], tools=MismatchTools())

    assert result.answer.warnings[0] == "Білок падає приблизно на 90%."  # the model's own
    assert result.answer.warnings[1] == (
        "Нутрієнти «молоко 2.5%» не збігаються зі специфікацією SPEC-001 в базі знань "
        "(protein_g 3.5 замість 2.8): перевірте вхідні дані розрахунку."
    )
    assert len(result.answer.warnings) == 2


async def test_no_warning_when_inputs_match_the_specs():
    result, _, _ = await run([*HAPPY, final()])
    assert result.answer.warnings == ["Білок падає приблизно на 90%."]
