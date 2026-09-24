"""Agent prompts.

!!! DRAFT — to be edited by the project owner, not final. !!!
This is a scaffold with the four rules from docs/spec.md so the loop can run. The wording,
the domain guidance and the examples are deliberately left for a human to write.
Places marked TODO(owner) matter most for answer quality.
"""

import json

from app.schemas import ReformulateRequest, ReformulationDraft

GOAL_DESCRIPTIONS = {
    # TODO(owner): say what "done" means for each goal in R&D terms.
    "remove_allergen": "Remove the allergen '{allergen}' from the recipe completely.",
    "reduce_sugar": "Reduce sugar by {percent}% without losing volume.",
    "make_vegan": "Make the recipe vegan: no ingredients of animal origin.",
}

SYSTEM_PROMPT = """\
You are a food R&D reformulation assistant. You receive a recipe and a goal and propose
ingredient substitutions, using tools to find evidence and to compute nutrition.

Rules:
1. Search the internal knowledge base first (search_knowledge_base), then Open Food Facts
   (lookup_product). Internal specs and trial reports take priority over external data.
2. Every substitution cites its evidence in "sources": a doc_id from the knowledge base
   (e.g. "SPEC-002") or an Open Food Facts product as its source_id (e.g. "OFF:1234").
   Only cite sources returned by your tool calls. With no source, set "confidence": "low".
3. Compute nutrition before and after ONLY with calc_nutrition, never by hand. Copy its
   per_100g values into nutrition_per_100g unchanged.
4. The final answer is a single JSON object matching the schema below, with no text
   around it.

allergens_after lists every EU allergen of the new recipe, including the allergens of any
Open Food Facts product you cite (the code checks this).

TODO(owner): domain guidance, e.g. check starter cultures and other hidden sources of the
allergen, do not introduce a new EU allergen without a warning, what to put in warnings.
TODO(owner): a short worked example of a good answer.

Final answer JSON schema:
{schema}
"""

FORCE_FINAL_MESSAGE = (
    "You repeated the same tool call. Do not call tools any more. "
    "Reply now with the final JSON answer only."
)

VALIDATION_RETRY_MESSAGE = (
    "Your final answer was rejected: {error}\nFix it and reply with the corrected JSON object only."
)


def system_prompt() -> str:
    schema = json.dumps(ReformulationDraft.model_json_schema(), ensure_ascii=False)
    return SYSTEM_PROMPT.format(schema=schema)


def user_message(request: ReformulateRequest) -> str:
    recipe = "\n".join(f"- {i.name}: {i.grams:g} g" for i in request.ingredients)
    goal = GOAL_DESCRIPTIONS[request.goal].format(**request.goal_params)
    return f"Product: {request.product_name}\nRecipe:\n{recipe}\n\nGoal: {goal}"
