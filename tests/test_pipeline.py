"""Fixed pipeline (AGENT_MODE=pipeline) with FakeLLM and fake tools: no network, no keys."""

import asyncio
import json

import pytest

from app.agent.loop import AgentError, AgentTimeoutError
from app.agent.pipeline import parse_nutrient_variants, run_pipeline
from app.agent.tools import CalcNutritionArgs, calc_nutrition
from app.llm.base import LLMError, LLMNotConfiguredError, Message
from app.llm.fake import FakeLLM
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient
from app.schemas import ReformulateRequest

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

N = {
    "milk": {"kcal": 52, "protein_g": 2.8, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7},
    "sugar": {"kcal": 400, "protein_g": 0, "fat_g": 0, "carbs_g": 100, "sugar_g": 100},
    "strawberry": {"kcal": 32, "protein_g": 0.7, "fat_g": 0.3, "carbs_g": 7.7, "sugar_g": 4.9},
    "starter": {"kcal": 58, "protein_g": 3.1, "fat_g": 2.4, "carbs_g": 5.2, "sugar_g": 3.9},
    "soy": {"kcal": 34, "protein_g": 3.3, "fat_g": 1.9, "carbs_g": 0.6, "sugar_g": 0.3},
    "dvs": {"kcal": 375, "protein_g": 4.0, "fat_g": 0.5, "carbs_g": 88, "sugar_g": 6.0},
}


def line(nutrients):
    return "; ".join(f"{k}: {v}" for k, v in nutrients.items())


def doc(doc_id, title, nutrients_text=None):
    item = {"doc_id": doc_id, "title": title, "text": f"{title}: …", "score": 0.9}
    if nutrients_text:
        item["nutrients_per_100g"] = nutrients_text
    return item


def variants(**named):  # SPEC-009 style: several products in one spec
    return " ".join(
        f"{name}: " + ", ".join(f"{k} {v}" for k, v in n.items()) + "." for name, n in named.items()
    )


STARTER_SPEC = doc(
    "SPEC-009", "Закваска", variants(**{"Варіант А": N["starter"], "Варіант Б": N["dvs"]})
)
SEARCH = {
    "goal": [doc("TRIAL-002", "Йогурт на сої"), STARTER_SPEC],
    "молоко 2.5%": [doc("SPEC-001", "Молоко 2.5%", line(N["milk"]))],
    "цукор": [doc("SPEC-006", "Цукор", line(N["sugar"]))],
    "полуниця заморожена": [doc("TRIAL-001", "Йогурт на кокосі")],
    "закваска": [STARTER_SPEC],
    "заміна для молоко 2.5%": [
        doc("SPEC-004", "Соєвий напій", line(N["soy"])),
        doc("TRIAL-002", "Йогурт на сої"),
    ],
}


class FakeTools:
    def __init__(self, slow_s: float = 0, off_found: bool = True) -> None:
        self.executed: list[tuple[str, dict]] = []
        self.slow_s, self.off_found = slow_s, off_found

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        await asyncio.sleep(self.slow_s)
        if name == "search_knowledge_base":
            query = arguments["query"]
            if query.startswith("заміна для"):
                key = query.split(",")[0]
                return {"results": SEARCH.get(key, [])}
            return {"results": SEARCH["goal" if ":" in query else query]}
        if name == "lookup_product":
            if not self.off_found:
                return {"found": False, "query": arguments["name"]}
            return {
                "found": True,
                "source_id": "OFF:777",
                "product_name": "Frozen strawberries",
                "nutrients_per_100g": N["strawberry"],
                "allergens": [],
            }
        if name == "calc_nutrition":
            return calc_nutrition(CalcNutritionArgs.model_validate(arguments).ingredients)
        raise AssertionError(name)


def choice(**overrides):
    base = {
        "substitutions": [
            {"original": "молоко 2.5%", "replacement": "соєвий напій", "grams": 800,
             "sources": ["SPEC-004", "TRIAL-002"]},
            {"original": "закваска", "replacement": "рослинна закваска DVS", "grams": 0.3,
             "sources": ["SPEC-009"]},
        ],
        "nutrients": {
            "молоко 2.5%": {"per_100g": N["milk"], "source": "SPEC-001"},
            "цукор": {"per_100g": N["sugar"], "source": "SPEC-006"},
            "полуниця заморожена": {"per_100g": None, "off_query": "frozen strawberries"},
            "закваска": {"per_100g": N["starter"], "source": "SPEC-009"},
            "соєвий напій": {"per_100g": N["soy"], "source": "SPEC-004"},
            "рослинна закваска DVS": {"per_100g": N["dvs"], "source": "SPEC-009"},
        },
    }  # fmt: skip
    return json.dumps({**base, **overrides}, ensure_ascii=False)


def per_100g(*lines):
    ingredients = [{"name": n, "grams": g, "nutrients_per_100g": N[k]} for n, g, k in lines]
    return calc_nutrition(CalcNutritionArgs(ingredients=ingredients).ingredients)["per_100g"]


BEFORE = per_100g(("молоко", 800, "milk"), ("цукор", 90, "sugar"),
                  ("полуниця", 100, "strawberry"), ("закваска", 10, "starter"))  # fmt: skip
AFTER = per_100g(("соя", 800, "soy"), ("цукор", 90, "sugar"),
                 ("полуниця", 100, "strawberry"), ("dvs", 0.3, "dvs"))  # fmt: skip


def final(before=BEFORE, **overrides) -> str:
    subs = json.loads(choice())["substitutions"]
    answer = {
        "substitutions": [{**s, "rationale": "Див. джерела.", "confidence": "high"} for s in subs],
        "allergens_before": ["milk"],
        "allergens_after": ["soybeans"],
        "nutrition_per_100g": {"before": before, "after": AFTER},
        "warnings": ["Новий алерген ЄС: соя."],
    }
    return json.dumps({**answer, **overrides}, ensure_ascii=False)


async def run(replies, tools=None, timeout_s=5.0):
    llm, tools = FakeLLM(replies), tools or FakeTools()
    result = await run_pipeline(llm, tools, REQUEST, timeout_s=timeout_s)
    return result, llm, tools


def calls(tools, name):
    return [args for n, args in tools.executed if n == name]


# ---------- happy path ----------


async def test_two_llm_calls_and_tools_called_by_code():
    result, llm, tools = await run([choice(), final()])

    assert len(llm.calls) == 2
    assert result.iterations == 2
    assert len([e for e in result.trace if e["type"] == "tool_call"]) >= 4
    assert len(calls(tools, "search_knowledge_base")) == 9  # goal + (own spec, replacement) x 4
    assert all(a["top_k"] == 3 for a in calls(tools, "search_knowledge_base"))
    assert calls(tools, "lookup_product") == [{"name": "frozen strawberries"}]
    assert len(calls(tools, "calc_nutrition")) == 2
    assert [e["purpose"] for e in result.trace if e["type"] == "llm_call"] == ["choose", "final"]
    assert result.trace[-1]["type"] == "final_answer"


async def test_replacement_searches_bring_replacement_specs_into_the_evidence():
    _, llm, tools = await run([choice(), final()])

    queries = [a["query"] for a in calls(tools, "search_knowledge_base")]
    assert "заміна для молоко 2.5%, без алергену milk" in queries
    assert "[SPEC-004] Соєвий напій | nutrients_per_100g: kcal: 34" in llm.calls[0][-1].content


async def test_recipes_for_calc_are_built_by_code():
    result, _, tools = await run([choice(), final()])

    before, after = (a["ingredients"] for a in calls(tools, "calc_nutrition"))
    assert [i["grams"] for i in before] == [800, 90, 100, 10]
    assert [i["name"] for i in after] == [
        "соєвий напій", "цукор", "полуниця заморожена", "рослинна закваска DVS"
    ]  # fmt: skip
    assert before[2]["nutrients_per_100g"] == N["strawberry"]  # from Open Food Facts
    assert result.answer.nutrition_per_100g.after.model_dump() == AFTER


# ---------- 1. numbers are checked against their source ----------


async def test_wrong_numbers_for_a_single_variant_spec_are_replaced_by_the_spec():
    # Live bug: the model gave coconut milk the numbers of cow milk.
    wrong = json.loads(choice())
    wrong["nutrients"]["соєвий напій"]["per_100g"] = N["milk"]
    _, _, tools = await run([json.dumps(wrong, ensure_ascii=False), final()])

    after = calls(tools, "calc_nutrition")[1]["ingredients"]
    assert after[0]["nutrients_per_100g"] == N["soy"]


async def test_multi_variant_spec_takes_the_matching_variant():
    _, _, tools = await run([choice(), final()])

    before, after = (a["ingredients"] for a in calls(tools, "calc_nutrition"))
    assert before[3]["nutrients_per_100g"] == N["starter"]  # SPEC-009 variant A
    assert after[3]["nutrients_per_100g"] == N["dvs"]  # SPEC-009 variant B


async def test_numbers_without_a_source_trigger_a_second_choice():
    unsourced = json.loads(choice())
    unsourced["nutrients"]["соєвий напій"] = {"per_100g": N["soy"], "source": None}
    result, llm, _ = await run([json.dumps(unsourced, ensure_ascii=False), choice(), final()])

    assert len(llm.calls) == 3  # choose, choose again, final
    assert "no verified nutrients per 100 g for: соєвий напій" in llm.calls[1][-1].content
    assert result.answer.nutrition_per_100g.after.model_dump() == AFTER


def test_parse_nutrient_variants():
    assert parse_nutrient_variants(line(N["milk"]) + " (уся лактоза)") == [
        {k: float(v) for k, v in N["milk"].items()}
    ]
    two = parse_nutrient_variants(STARTER_SPEC["nutrients_per_100g"])
    assert [v["kcal"] for v in two] == [58, 375]
    assert parse_nutrient_variants("kcal: 20 (0.2–0.4 ккал/г)") == []  # incomplete set


# ---------- 2-3. evidence and choice shape ----------


async def test_sources_outside_the_evidence_trigger_a_second_choice():
    bad = json.loads(choice())
    bad["substitutions"][0]["sources"] = ["SPEC-002"]  # mentioned in texts, not retrieved
    _, llm, _ = await run([json.dumps(bad, ensure_ascii=False), choice(), final()])

    assert "sources ['SPEC-002'] are not in the evidence" in llm.calls[1][-1].content


async def test_combined_ingredients_are_rejected_and_rechosen():
    combo = json.loads(choice())
    combo["substitutions"][0]["replacement"] = "цукор 63 г + ER-ST 27 г"
    _, llm, _ = await run([json.dumps(combo, ensure_ascii=False), choice(), final()])

    assert "one ingredient per substitution" in llm.calls[1][-1].content


async def test_added_ingredient_with_original_null_is_appended():
    added = json.loads(choice())
    added["substitutions"].append(
        {"original": None, "replacement": "соєвий напій", "grams": 5, "sources": ["SPEC-004"]}
    )
    after_added = per_100g(("соя", 800, "soy"), ("цукор", 90, "sugar"),
                           ("полуниця", 100, "strawberry"), ("dvs", 0.3, "dvs"),
                           ("соя", 5, "soy"))  # fmt: skip
    answer = final(nutrition_per_100g={"before": BEFORE, "after": after_added})
    _, _, tools = await run([json.dumps(added, ensure_ascii=False), answer])

    after = calls(tools, "calc_nutrition")[1]["ingredients"]
    assert [i["name"] for i in after][-1] == "соєвий напій"
    assert after[-1]["grams"] == 5


# ---------- 4. missing data and the LLM call budget ----------


async def test_missing_data_after_the_second_choice_stops_before_the_final_call():
    no_off = FakeTools(off_found=False)  # strawberries: no spec and nothing in OFF
    with pytest.raises(AgentError) as err:
        await run([choice(), choice()], tools=no_off)

    assert err.value.code == "agent_missing_data"
    assert "полуниця заморожена" in err.value.message
    assert [e["purpose"] for e in err.value.trace if e["type"] == "llm_call"] == [
        "choose",
        "choose",
    ]


async def test_invalid_final_answer_is_retried_once_then_502():
    with pytest.raises(AgentError) as err:
        await run([choice(), "not json", '{"substitutions": []}'])

    assert err.value.code == "agent_invalid_output"
    assert [e["type"] for e in err.value.trace].count("llm_call") == 3


async def test_after_a_second_choice_the_final_answer_is_not_retried():
    unsourced = json.loads(choice())
    unsourced["nutrients"]["цукор"]["source"] = None
    with pytest.raises(AgentError) as err:
        await run([json.dumps(unsourced, ensure_ascii=False), choice(), "not json", final()])

    assert err.value.code == "agent_invalid_output"
    assert [e["type"] for e in err.value.trace].count("llm_call") == 3  # budget kept


async def test_invalid_final_answer_fixed_on_retry():
    made_up = final(before={"kcal": 78})  # not from calc_nutrition
    result, llm, _ = await run([choice(), made_up, final()])

    assert len(llm.calls) == 3
    assert "rejected" in llm.calls[2][-1].content
    assert result.answer.nutrition_per_100g.before.model_dump() == BEFORE


async def test_invalid_choice_twice_is_502():
    with pytest.raises(AgentError) as err:
        await run(["not json", "still not json"])

    assert err.value.code == "agent_missing_data"
    assert "invalid choice" in err.value.message


# ---------- failures ----------


async def test_llm_error_keeps_the_trace_of_tool_calls():
    def fail(_messages):
        raise LLMError("overloaded", "llm_error", 502)

    with pytest.raises(AgentError) as err:
        await run([fail])

    assert err.value.code == "llm_error"
    assert len([e for e in err.value.trace if e["type"] == "tool_call"]) == 9


async def test_timeout_applies_to_the_pipeline():
    with pytest.raises(AgentTimeoutError):
        await run([choice(), final()], FakeTools(slow_s=0.5), timeout_s=0.2)


# ---------- 5. API keys with stray characters ----------


@pytest.mark.parametrize(
    ("client", "env"),
    [(GroqClient, "GROQ_API_KEY"), (GeminiClient, "GEMINI_API_KEY")],
)
async def test_api_key_with_a_non_ascii_character_gives_a_clear_error(client, env):
    llm = client("тgsk_abcdef", "model")  # Cyrillic "т" typed with the wrong keyboard layout

    with pytest.raises(LLMNotConfiguredError) as err:
        await llm.complete([Message("user", "x")])

    assert err.value.status_code == 503
    assert f"{env} contains an invalid character at position 0" in err.value.message
    assert "CYRILLIC SMALL LETTER TE" in err.value.message


async def test_original_ingredient_not_mapped_by_the_model_uses_its_own_spec():
    # Live: the model left "молоко 2.5%" without a source; its own search had found SPEC-001.
    unmapped = json.loads(choice())
    del unmapped["nutrients"]["молоко 2.5%"]
    _, llm, tools = await run([json.dumps(unmapped, ensure_ascii=False), final()])

    assert len(llm.calls) == 2  # no second choice needed
    before = calls(tools, "calc_nutrition")[0]["ingredients"]
    assert before[0]["nutrients_per_100g"] == N["milk"]
    assert "- молоко 2.5% -> [SPEC-001]" in llm.calls[0][-1].content
    assert "- полуниця заморожена -> no spec" in llm.calls[0][-1].content


async def test_the_choice_is_recorded_in_the_trace():
    result, _, _ = await run([choice(), final()])

    entry = next(e for e in result.trace if e["type"] == "choice")
    assert (
        entry["substitutions"][0] == "молоко 2.5% -> соєвий напій (800 g) ['SPEC-004', 'TRIAL-002']"
    )
    assert entry["nutrient_sources"]["полуниця заморожена"] == "frozen strawberries"


async def test_original_with_a_multi_variant_spec_takes_the_first_variant_as_an_assumption():
    # Live: the model never mapped the original "закваска"; SPEC-009 has variants A and B.
    unmapped = json.loads(choice())
    del unmapped["nutrients"]["закваска"]
    result, llm, tools = await run([json.dumps(unmapped, ensure_ascii=False), final()])

    assert len(llm.calls) == 2
    before = calls(tools, "calc_nutrition")[0]["ingredients"]
    assert before[3]["nutrients_per_100g"] == N["starter"]  # variant A, the dairy starter
    assumption = next(e for e in result.trace if e["type"] == "assumption")
    assert assumption["ingredient"] == "закваска"
    assert assumption["note"] == "nutrients from SPEC-009, variant 1 of 2"


async def test_second_choice_lists_the_specs_that_have_nutrients():
    no_data = json.loads(choice())
    no_data["nutrients"]["соєвий напій"] = {"per_100g": None, "source": None}
    _, llm, _ = await run([json.dumps(no_data, ensure_ascii=False), choice(), final()])

    feedback = llm.calls[1][-1].content
    assert "Specs with nutrients:" in feedback
    assert "[SPEC-004] Соєвий напій" in feedback


async def test_off_query_must_be_english():
    ukrainian = json.loads(choice())
    ukrainian["nutrients"]["полуниця заморожена"]["off_query"] = "заморожена полуниця"
    _, llm, tools = await run([json.dumps(ukrainian, ensure_ascii=False), choice(), final()])

    assert "off_query must be an English product name" in llm.calls[1][-1].content
    assert {"name": "заморожена полуниця"} not in calls(tools, "lookup_product")


def test_reduce_sugar_searches_for_sweeteners():
    from app.agent import prompts

    request = REQUEST.model_copy(update={"goal": "reduce_sugar", "goal_params": {"percent": 30}})
    query = prompts.pipeline_replacement_query(request, "цукор")
    assert (
        query
        == "заміна для цукор, замінник цукру: поліоли, інтенсивні підсолоджувачі, рідкісні цукри"
    )
    assert "sugar replacement is a sweetener" in prompts.PIPELINE_CHOOSE_PROMPT
