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
from app.retrieval import search_chunks

logger = logging.getLogger(__name__)

NUTRIENTS = ("kcal", "protein_g", "fat_g", "carbs_g", "sugar_g")

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
    top_k: int = Field(default=5, ge=1, le=10)


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

_NUTRIENTS_SCHEMA = {
    "type": "object",
    "description": "Nutrients per 100 g of this ingredient",
    "properties": {n: {"type": "number"} for n in NUTRIENTS},
}

TOOL_SPECS = [
    ToolSpec(
        name="search_knowledge_base",
        description=(
            "Search internal R&D documents (ingredient specs, trial reports, guidelines) "
            "by meaning. Use first: internal documents take priority over external data. "
            "Returns chunks with doc_id, title, text and similarity score."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_product",
        description=(
            "Look up a food product in Open Food Facts by name. Returns nutrients per 100 g, "
            "allergens and ingredients of the first matching product with nutrition data. "
            "Use when the knowledge base has no data for an ingredient. "
            "Prefer English product names: Open Food Facts data is mostly English/French."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Product name, e.g. 'coconut milk'"}
            },
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="calc_nutrition",
        description=(
            "Compute kcal, protein_g, fat_g, carbs_g, sugar_g per 100 g of the finished "
            "product as the mass-weighted average of its ingredients. Always use this "
            "for nutrition numbers, never calculate them yourself."
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

    async def search_knowledge_base(self, query: str, top_k: int = 5) -> dict[str, Any]:
        chunks = await search_chunks(self._pool, self._embedder, query, top_k)
        return {
            "results": [
                {"doc_id": c.doc_id, "title": c.title, "text": c.text, "score": c.score}
                for c in chunks
            ]
        }

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


def _nutrients(product: dict[str, Any]) -> dict[str, float]:
    nutriments = product.get("nutriments") or {}
    return {
        ours: round(float(nutriments[off]), 2)
        for ours, off in _OFF_NUTRIMENTS.items()
        if isinstance(nutriments.get(off), int | float)
    }


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
        for e in exc.errors()
    )
