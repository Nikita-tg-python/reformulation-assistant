"""The agent's three tools: JSON schemas for tool calling plus their implementations.

Every tool returns a JSON-serialisable dict. Expected failures (bad arguments, network
errors, product not found) come back as data the model can read and react to, never as
exceptions that would abort the agent run.
"""

import logging
import re
from typing import Any

import asyncpg
import httpx
from pydantic import BaseModel, Field, ValidationError

from app.embeddings import Embedder
from app.llm.base import ToolSpec
from app.retrieval import search_chunks, spec_nutrients

logger = logging.getLogger(__name__)

NUTRIENTS = ("kcal", "protein_g", "fat_g", "carbs_g", "sugar_g")

# Chunk text sent back to the model is cut to keep the history inside free-tier token limits
# (Groq: 8000 tokens/min). The trace keeps the full call; only the model's copy is shortened.
KB_TEXT_LIMIT = 500
KB_DEFAULT_TOP_K = 3
# 3 distinct documents carry what 5 chunks used to (those were 3 docs with duplicates),
# at ~60% of the tokens. The model asked for the maximum every time, so the cap matters.
KB_MAX_TOP_K = 3
KB_CHUNKS_PER_RESULT = 3  # chunks fetched per requested document before de-duplication

OFF_SEARCH_URL = "https://search.openfoodfacts.org/search"  # Open Food Facts full-text search
OFF_TIMEOUT_S = 5.0
OFF_USER_AGENT = "reformulation-assistant/0.1 (learning project)"  # OFF requires a named UA
OFF_CANDIDATES = 10  # search is fuzzy: look at several hits to find a relevant one
_OFF_NUTRIMENTS = {
    "kcal": "energy-kcal_100g",
    "protein_g": "proteins_100g",
    "fat_g": "fat_100g",
    "carbs_g": "carbohydrates_100g",
    "sugar_g": "sugars_100g",
}
# OFF allergen tags -> the EU codes used across the service (see GUIDE-001).
_OFF_ALLERGENS = {
    "gluten": "gluten",
    "crustaceans": "crustaceans",
    "eggs": "eggs",
    "fish": "fish",
    "peanuts": "peanuts",
    "soybeans": "soybeans",
    "milk": "milk",
    "nuts": "nuts",
    "celery": "celery",
    "mustard": "mustard",
    "sesame-seeds": "sesame",
    "sulphur-dioxide-and-sulphites": "sulphites",
    "lupin": "lupin",
    "molluscs": "molluscs",
}


# ---------- argument models (validate what the model sends) ----------


class SearchKnowledgeBaseArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=KB_DEFAULT_TOP_K, ge=1, le=KB_MAX_TOP_K)


class LookupProductArgs(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class NutritionIngredient(BaseModel):
    name: str = Field(min_length=1)
    grams: float = Field(ge=0)
    # Missing keys or null values are allowed: they are reported, not guessed.
    nutrients_per_100g: dict[str, float | None] = Field(default_factory=dict)


class CalcNutritionArgs(BaseModel):
    ingredients: list[NutritionIngredient] = Field(max_length=100)


# ---------- JSON schemas for tool calling ----------

# Descriptions are kept short on purpose: the schemas are resent with every LLM call.
# null is allowed, as in NutritionIngredient: Open Food Facts often lacks a value (e.g. kcal),
# and Groq rejects the whole tool call when the arguments break this schema.
_NUTRIENTS_SCHEMA = {
    "type": "object",
    "properties": {n: {"type": ["number", "null"]} for n in NUTRIENTS},
}

TOOL_SPECS = [
    ToolSpec(
        name="search_knowledge_base",
        description=(
            "Search internal R&D docs (specs, trial reports, guidelines). Use first. "
            "Returns up to top_k documents: doc_id, title, best-matching text, score, and "
            "for ingredient specs nutrients_per_100g: use these numbers in calc_nutrition."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": KB_MAX_TOP_K,
                    "default": KB_DEFAULT_TOP_K,
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_product",
        description=(
            "Open Food Facts product by English name: nutrients per 100 g, allergens. "
            "Only when the knowledge base has no data."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="calc_nutrition",
        description=(
            "Nutrition per 100 g of the finished product (mass-weighted average). "
            "Always use it for nutrition numbers."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "grams": {"type": "number", "minimum": 0},
                            "nutrients_per_100g": _NUTRIENTS_SCHEMA,
                        },
                        "required": ["name", "grams", "nutrients_per_100g"],
                    },
                }
            },
            "required": ["ingredients"],
        },
    ),
]


# ---------- calc_nutrition: pure arithmetic ----------


def calc_nutrition(ingredients: list[NutritionIngredient]) -> dict[str, Any]:
    """Mass-weighted average per 100 g of the finished product.

    A missing (or null) nutrient counts as 0 for that ingredient and is listed in
    `missing`, so the result is an honest lower bound instead of a crash or a guess.
    Zero total mass gives null values plus a warning.
    """
    total = sum(i.grams for i in ingredients)
    missing = [
        {"ingredient": i.name, "nutrients": gaps}
        for i in ingredients
        if i.grams > 0 and (gaps := [n for n in NUTRIENTS if i.nutrients_per_100g.get(n) is None])
    ]
    warnings = []
    if total <= 0:
        warnings.append("total mass is 0 g: nutrition per 100 g is undefined")
        per_100g: dict[str, float | None] = dict.fromkeys(NUTRIENTS)
    else:
        per_100g = {
            n: round(
                sum(i.grams * (i.nutrients_per_100g.get(n) or 0.0) for i in ingredients) / total,
                1,
            )
            for n in NUTRIENTS
        }
    if missing:
        warnings.append("some nutrients are missing and counted as 0: values are lower bounds")
    return {
        "total_grams": round(total, 1),
        "per_100g": per_100g,
        "missing": missing,
        "warnings": warnings,
    }


# ---------- per-run toolbox ----------


class AgentTools:
    """Tools bound to one agent run: shared DB/embedder, per-run Open Food Facts cache."""

    def __init__(
        self, pool: asyncpg.Pool, embedder: Embedder, http: httpx.AsyncClient | None = None
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._http = http
        self._product_cache: dict[str, dict[str, Any]] = {}

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool by name. Unknown tools and invalid arguments come back as errors."""
        try:
            if name == "search_knowledge_base":
                args = SearchKnowledgeBaseArgs.model_validate(arguments)
                return await self.search_knowledge_base(args.query, args.top_k)
            if name == "lookup_product":
                return await self.lookup_product(LookupProductArgs.model_validate(arguments).name)
            if name == "calc_nutrition":
                return calc_nutrition(CalcNutritionArgs.model_validate(arguments).ingredients)
        except ValidationError as exc:
            return _error("invalid_arguments", _describe(exc))
        return _error("unknown_tool", f"no tool named {name!r}")

    async def search_knowledge_base(
        self, query: str, top_k: int = KB_DEFAULT_TOP_K
    ) -> dict[str, Any]:
        # One result per document: a second chunk of the same doc costs ~200 tokens and
        # rarely adds anything. Fetch extra chunks so top_k still means top_k documents.
        chunks = await search_chunks(
            self._pool, self._embedder, query, top_k * KB_CHUNKS_PER_RESULT
        )
        best: dict[str, Any] = {}
        for c in chunks:  # already ordered by score
            if c.doc_id not in best and len(best) < top_k:
                best[c.doc_id] = c
        nutrients = await spec_nutrients(self._pool, sorted(best))
        results = []
        for c in best.values():
            item = {
                "doc_id": c.doc_id,
                "title": c.title,
                "text": _shorten(c.text, KB_TEXT_LIMIT),
                "score": c.score,
            }
            if c.doc_id in nutrients:
                item["nutrients_per_100g"] = nutrients[c.doc_id]
            results.append(item)
        return {"results": results}

    async def lookup_product(self, name: str) -> dict[str, Any]:
        key = " ".join(name.lower().split())
        if key in self._product_cache:
            return self._product_cache[key]
        try:
            result = await self._fetch_product(name)
        except httpx.TimeoutException:
            return _error("timeout", f"Open Food Facts did not answer in {OFF_TIMEOUT_S:g} s")
        except httpx.HTTPStatusError as exc:
            return _error("http_error", f"Open Food Facts returned {exc.response.status_code}")
        except (httpx.HTTPError, ValueError) as exc:  # connection errors, malformed JSON
            return _error("network_error", f"{type(exc).__name__}: {exc}")
        self._product_cache[key] = result  # errors are not cached: a retry may succeed
        return result

    async def _fetch_product(self, name: str) -> dict[str, Any]:
        params = {
            "q": name,
            "page_size": OFF_CANDIDATES,
            "fields": "code,product_name,nutriments,allergens_tags,ingredients_text",
        }
        headers = {"User-Agent": OFF_USER_AGENT}
        if self._http is None:
            async with httpx.AsyncClient(timeout=OFF_TIMEOUT_S) as client:
                response = await client.get(OFF_SEARCH_URL, params=params, headers=headers)
        else:
            response = await self._http.get(
                OFF_SEARCH_URL, params=params, headers=headers, timeout=OFF_TIMEOUT_S
            )
        response.raise_for_status()
        hits = response.json().get("hits") or []
        # OFF search is fuzzy and returns something for any query ("zzqxw 123" finds a
        # product named "123"), so keep only hits whose name matches the query.
        relevant = [h for h in hits if _name_matches(name, h.get("product_name") or "")]
        if not relevant:
            return {
                "found": False,
                "source": "openfoodfacts",
                "query": name,
                "message": "no product whose name matches the query; try an English name",
            }
        # First relevant hit with nutrition data; otherwise the first relevant hit.
        product = next((h for h in relevant if _nutrients(h)), relevant[0])
        tags = product.get("allergens_tags") or []
        return {
            "found": True,
            "source": "openfoodfacts",
            "source_id": f"OFF:{product.get('code')}",  # what the agent cites in sources
            "code": product.get("code"),
            "product_name": product.get("product_name"),
            "url": f"https://world.openfoodfacts.org/product/{product.get('code')}",
            "nutrients_per_100g": _nutrients(product),
            "allergens": sorted(
                {
                    _OFF_ALLERGENS[t.split(":", 1)[-1]]
                    for t in tags
                    if t.split(":", 1)[-1] in _OFF_ALLERGENS
                }
            ),
            "allergen_tags": tags,
            "ingredients_text": product.get("ingredients_text"),
        }


def _name_matches(query: str, product_name: str) -> bool:
    """Most of the query's significant words occur in the product name.

    Words are compared by a crude stem (drop up to 3 trailing letters, keep at least 4),
    so "strawberries" matches "strawberry" but "кокосове молоко" does not match "Молоко".
    """
    words = re.findall(r"\w+", query.lower())
    words = [w for w in words if len(w) >= 3] or words
    name = product_name.lower()
    matched = sum(w[: max(4, len(w) - 3)] in name for w in words)
    return 2 * matched > len(words)


def compact_result(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    """What the model keeps of a tool result after it has seen it once in full.

    Older results are resent on every later call; the full chunk text is what blows the
    token budget. Sources stay citable (doc_id, source_id) and numbers stay available.
    """
    if "error" in result:
        return result
    if tool == "search_knowledge_base":
        return {
            "results": [
                {k: r[k] for k in ("doc_id", "title", "score", "nutrients_per_100g") if k in r}
                for r in result.get("results", [])
            ],
            "note": "text shown earlier, omitted",
        }
    if tool == "lookup_product" and result.get("found"):
        keep = ("source_id", "product_name", "nutrients_per_100g", "allergens")
        return {"found": True, **{k: result.get(k) for k in keep}}
    if tool == "calc_nutrition":
        return {"per_100g": result.get("per_100g"), "warnings": result.get("warnings")}
    return result


def _nutrients(product: dict[str, Any]) -> dict[str, float]:
    nutriments = product.get("nutriments") or {}
    return {
        ours: round(float(nutriments[off]), 2)
        for ours, off in _OFF_NUTRIMENTS.items()
        if isinstance(nutriments.get(off), int | float)
    }


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
        for e in exc.errors()
    )
