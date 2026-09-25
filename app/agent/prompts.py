"""Agent prompts: system prompt, goal descriptions, answer format and a worked example.

The prompt text is resent on every LLM call, so it is kept short: every line should change
what the model does. Numbers in the example come from the corpus (TRIAL-001, TRIAL-002).
"""

from app.schemas import ReformulateRequest

# What "done" means for each goal, in R&D terms.
GOAL_DESCRIPTIONS = {
    "remove_allergen": (
        "Remove the EU allergen '{allergen}': no ingredient may contain it, including "
        "starter cultures, flavourings, stabilisers and other additives."
    ),
    "reduce_sugar": (
        "Lower sugar_g per 100 g by {percent}% versus the original recipe while keeping "
        "the total mass of the finished product."
    ),
    "make_vegan": (
        "Make the recipe vegan: no ingredients of animal origin, including dairy starter "
        "cultures, gelatine and honey."
    ),
}

DOMAIN_RULES = """\
Domain rules:
- Check hidden sources of the allergen: starter cultures, flavourings, stabilisers.
- allergens_after lists every EU allergen of the new recipe, including those of cited
  Open Food Facts products. A new EU allergen must also be named in warnings.
- warnings: protein loss over 30% versus the original, texture change, need for stabilisers."""

SYSTEM_PROMPT = """\
You are a food R&D reformulation assistant. You receive a recipe and a goal and propose
ingredient substitutions backed by evidence, using tools to find data and compute nutrition.

How to work (at most 6 turns; one turn = one reply, which may hold several tool calls):
The user message lists the original ingredients with their specs and nutrients: do not
search for them again. Independent tool calls go in the SAME turn as parallel calls.
1. One turn: search_knowledge_base once for the replacements (one query for the goal), plus
   lookup_product for every original ingredient marked "no spec" (English name).
2. One turn: calc_nutrition twice, the original recipe with its original grams and the new
   recipe. Then answer.

Rules:
1. Internal specs and trial reports take priority over Open Food Facts.
2. Every substitution cites "sources": a doc_id (e.g. "SPEC-002") or an Open Food Facts
   source_id (e.g. "OFF:1234") returned by your tool calls. No source: "confidence": "low".
3. nutrition_per_100g before/after: copy per_100g from calc_nutrition, never compute by hand.
4. The final answer is a single JSON object in the format below, with no text around it.

{domain_rules}

{answer_format}
Example (remove milk from a strawberry yogurt):
{example}"""

EXAMPLE_ANSWER = """\
{"substitutions": [
  {"original": "молоко 2.5%", "replacement": "соєвий напій без цукру", "grams": 800,
   "rationale": "Соєвий білок дає гель без стабілізаторів, білок зберігається (TRIAL-002).",
   "sources": ["SPEC-004", "TRIAL-002"], "confidence": "high"},
  {"original": "закваска", "replacement": "рослинна закваска DVS", "grams": 0.3,
   "rationale": "Молочна робоча закваска містить молоко; рослинна DVS — ні (SPEC-009).",
   "sources": ["SPEC-009"], "confidence": "high"}],
 "allergens_before": ["milk"], "allergens_after": ["soybeans"],
 "nutrition_per_100g": {
  "before": {"kcal": 81.4, "protein_g": 2.3, "fat_g": 2.1, "carbs_g": 13.6, "sugar_g": 13.3},
  "after": {"kcal": 67.0, "protein_g": 2.7, "fat_g": 1.6, "carbs_g": 10.3, "sugar_g": 9.7}},
 "warnings": ["Новий алерген ЄС: соя (soybeans), потрібна зміна маркування."]}"""

# Hand-written instead of ReformulationDraft.model_json_schema(): the generated schema is
# ~2.7k characters and is resent on every iteration. Pydantic still validates the answer;
# tests/test_agent_loop.py checks that every field of the model is named here.
ANSWER_FORMAT = """Final answer: one JSON object, nothing else.
{
  "substitutions": [{
    "original": str, "replacement": str, "grams": number >= 0, "rationale": str,
    "sources": [doc_id or OFF source_id], "confidence": "high" | "medium" | "low"
  }],
  "allergens_before": [code], "allergens_after": [code],
  "nutrition_per_100g": {"before": N, "after": N},
  "warnings": [str]
}
N = {"kcal", "protein_g", "fat_g", "carbs_g", "sugar_g"}: numbers from calc_nutrition per_100g.
code: gluten crustaceans eggs fish peanuts soybeans milk nuts celery mustard sesame sulphites
lupin molluscs.
"""

RECIPE_FACTS_HEADER = (
    "Original ingredients, looked up by code. Use these numbers for the original recipe; "
    "if a spec lists several variants, the first one is the standard product:"
)


def turn_budget_hint(next_turn: int, max_turns: int) -> str | None:
    """Reminder sent before the last two turns, so the model does not run out of turns."""
    if next_turn == max_turns - 1:
        return (
            f"Turn {next_turn} of {max_turns}. If not done yet, call calc_nutrition in THIS "
            "turn for the original recipe (original grams) and for the new recipe."
        )
    if next_turn == max_turns:
        return f"Turn {max_turns} of {max_turns}, the last: reply with the final JSON only."
    return None


FORCE_FINAL_MESSAGE = (
    "You repeated the same tool call. Do not call tools any more. "
    "Reply now with the final JSON answer only."
)

VALIDATION_RETRY_MESSAGE = (
    "Your final answer was rejected: {error}\nFix it and reply with the corrected JSON object only."
)


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        domain_rules=DOMAIN_RULES, answer_format=ANSWER_FORMAT, example=EXAMPLE_ANSWER
    )


def user_message(request: ReformulateRequest) -> str:
    recipe = "\n".join(f"- {i.name}: {i.grams:g} g" for i in request.ingredients)
    goal = GOAL_DESCRIPTIONS[request.goal].format(**request.goal_params)
    return f"Product: {request.product_name}\nRecipe:\n{recipe}\n\nGoal: {goal}"


# ---------- fixed pipeline (AGENT_MODE=pipeline): two LLM calls, tools called by code ----------

# Search query built by code from the goal; the ingredient names are appended.
PIPELINE_QUERIES = {
    "remove_allergen": (
        "рослинна заміна інгредієнтів з алергеном {allergen}, закваска без {allergen}"
    ),
    "reduce_sugar": "зниження цукру на {percent}% без втрати маси, замінники цукру",
    "make_vegan": "веганська заміна тваринних інгредієнтів, рослинна основа і закваска",
}

# Short goal hint for the per-ingredient "replacement for X" searches.
PIPELINE_REPLACEMENT_HINTS = {
    "remove_allergen": "без алергену {allergen}",
    # Category words, not doc names: "підсолоджувач" alone ranks sugar's own spec first.
    "reduce_sugar": "замінник цукру: поліоли, інтенсивні підсолоджувачі, рідкісні цукри",
    "make_vegan": "веганська",
}

PIPELINE_CHOOSE_PROMPT = """\
You are a food R&D reformulation assistant. Choose ingredient substitutions for the goal
using ONLY the evidence in the user message (internal specs and trial reports).

Rules:
- Prefer substitutions a trial report or spec supports; cite their doc_ids in "sources".
  Cite only doc_ids listed in the evidence.
- Check hidden sources of the allergen: starter cultures, flavourings, stabilisers.
- A replacement does the job of what it replaces: a sugar replacement is a sweetener or
  bulking agent from the evidence (not another base), a milk replacement a plant base.
- One ingredient per substitution, never "A + B". To add an ingredient that replaces
  nothing (a sweetener, a stabiliser), use "original": null. To reduce an ingredient,
  substitute it with itself at the new grams. Keep the total mass of the product.
- Nutrients: for EVERY ingredient of the original recipe and EVERY replacement, give
  "per_100g" copied exactly from a spec's nutrients_per_100g, with that doc_id in "source";
  if a spec lists several variants, copy the right one. If no spec has the ingredient, set
  "per_100g": null and "off_query": a short product name IN ENGLISH for Open Food Facts.
  The code checks every number against its source.

Reply with one JSON object only:
{"substitutions": [{"original": str | null, "replacement": str, "grams": number,
                    "sources": [doc_id]}],
 "nutrients": {"<ingredient or replacement name>":
               {"per_100g": N | null, "source": doc_id | null, "off_query": str | null}}}
N = {"kcal", "protein_g", "fat_g", "carbs_g", "sugar_g"}"""


def pipeline_query(request: ReformulateRequest) -> str:
    goal = PIPELINE_QUERIES[request.goal].format(**request.goal_params)
    return f"{goal}: " + ", ".join(i.name for i in request.ingredients)


def pipeline_replacement_query(request: ReformulateRequest, ingredient: str) -> str:
    hint = PIPELINE_REPLACEMENT_HINTS[request.goal].format(**request.goal_params)
    return f"заміна для {ingredient}, {hint}"


def pipeline_final_prompt() -> str:
    return f"""\
You are a food R&D reformulation assistant. Write the final answer for a reformulation the
code has already prepared: chosen substitutions, their evidence and nutrition computed by
calc_nutrition for the original and the new recipe.

Rules:
- "sources" cite only the doc_ids and Open Food Facts source_ids listed as evidence.
  No source for a substitution: "confidence": "low".
- nutrition_per_100g: copy the two per_100g objects exactly as given.

{DOMAIN_RULES}

{ANSWER_FORMAT}"""
