"""Structured facts of ingredient specs: nutrients per variant and EU allergen codes.

Parsed from the markdown at ingest time (documents.nutrients, documents.allergens), so code
can check numbers instead of reading them from text. Formats in data/corpus/:

    ## Нутрієнти на 100 г                    ## Алергени (перелік ЄС)
    - kcal: 52                               Молоко ... (Регламент ЄС 1169/2011, додаток II,
    - protein_g: 2.8 (note)                  пункт 7). Інших алергенів немає.
    or per variant, one paragraph each:
    Аквафаба: kcal 18, protein_g 1.0, ...    - SG-3: алергенів із переліку ЄС немає.
"""

import re
from typing import Any

NUTRIENTS = ("kcal", "protein_g", "fat_g", "carbs_g", "sugar_g")
TOLERANCE = 0.05  # same as the before/after check in the agent loop

# Annex II of Regulation (EU) 1169/2011, item number -> the codes used in the API.
EU_ALLERGENS = {
    1: "gluten", 2: "crustaceans", 3: "eggs", 4: "fish", 5: "peanuts", 6: "soybeans",
    7: "milk", 8: "nuts", 9: "celery", 10: "mustard", 11: "sesame", 12: "sulphites",
    13: "lupin", 14: "molluscs",
}  # fmt: skip

_SECTION = r"^## {title}[^\n]*\n(.*?)(?=^## |\Z)"
_NUMBER = re.compile(r"\b(kcal|protein_g|fat_g|carbs_g|sugar_g)\b\s*:?\s*(-?\d+(?:[.,]\d+)?)")
# The spec cites Annex II items: "пункт 7". Counting item numbers and not words avoids
# false hits like "Кокос не належить до горіхів".
_ANNEX_ITEM = re.compile(r"пункт\w*\s+(\d{1,2})\b")


def _section(content: str, title: str) -> str | None:
    match = re.search(_SECTION.format(title=title), content, re.M | re.S)
    return match.group(1) if match else None


def parse_nutrients(content: str) -> list[dict[str, Any]] | None:
    """Variants from "## Нутрієнти на 100 г": [{"variant": name or None, "per_100g": {...}}].

    One variant per paragraph (a bullet list is one paragraph). The text before the first
    number is the variant name ("Аквафаба:", "Варіант Б (сухий порошок):"). Paragraphs
    without all five nutrients are notes, not data. None when the section is missing.
    """
    section = _section(content, "Нутрієнти на 100 г")
    if section is None:
        return None
    variants = []
    for paragraph in re.split(r"\n\s*\n", section.strip()):
        values: dict[str, float] = {}
        for name, value in _NUMBER.findall(paragraph):
            values.setdefault(name, float(value.replace(",", ".")))
        if set(values) != set(NUTRIENTS):
            continue
        label = paragraph[: _NUMBER.search(paragraph).start()]
        label = re.sub(r"\s+на 100 г\s*$", "", label.strip().rstrip(":").strip()).strip("- ")
        variants.append({"variant": label or None, "per_100g": {n: values[n] for n in NUTRIENTS}})
    return variants


def parse_allergens(content: str) -> list[str] | None:
    """EU allergen codes cited in "## Алергени": all variants together (a union), so a
    multi-variant spec lists every allergen any variant has. None when the section is missing.
    """
    section = _section(content, "Алергени")
    if section is None:
        return None
    items = {int(n) for n in _ANNEX_ITEM.findall(section)}
    return [EU_ALLERGENS[n] for n in sorted(items) if n in EU_ALLERGENS]


def _words(text: str) -> set[str]:
    """Lowercase words; "2.5%" stays one word, trailing punctuation is dropped."""
    tokens = re.findall(r"[\w%.'-]+", text.lower().replace("’", "'"))
    return {t.strip(".-") for t in tokens} - {""}


def own_spec(name: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The spec of an ingredient among search results: the title must contain every word of
    the name, and the closest title wins (most of its words covered; ties keep search order).

    The first result with nutrients is not enough: hybrid search ranks "Кокосове молоко"
    first for "закваска" (its text mentions закваска), and the recipe got coconut numbers.
    "молоко" matches both SPEC-001 "Молоко коров'яче 2.5% жиру" and SPEC-002 "Кокосове
    молоко ...": coverage 1/4 beats 1/6, so plain milk is cow's milk.
    """
    words = _words(name)
    best, best_cover = None, 0.0
    for r in results:
        title = _words(r.get("title", ""))
        if not r.get("nutrients_per_100g") or not words or not words <= title:
            continue
        cover = len(words) / len(title)
        if cover > best_cover:
            best, best_cover = r, cover
    return best


def spec_mismatches(
    ingredients: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calc_nutrition inputs that contradict a spec in the knowledge base.

    An ingredient matches a spec when all words of its name occur in the spec title (then
    every variant is a candidate) or in a variant name (then only that variant). Numbers
    must agree within TOLERANCE with at least one candidate, for every nutrient given.
    Names that match no spec (e.g. English names from Open Food Facts) are not checked.
    """
    found = []
    for ingredient in ingredients:
        words = _words(str(ingredient.get("name", "")))
        given = {
            n: float(v)
            for n, v in (ingredient.get("nutrients_per_100g") or {}).items()
            if n in NUTRIENTS and v is not None
        }
        if not words or not given:
            continue
        candidates = []  # (doc_id, variant)
        for spec in specs:
            title_match = words <= _words(spec["title"])
            for variant in spec["nutrients"]:
                if title_match or (variant["variant"] and words <= _words(variant["variant"])):
                    candidates.append((spec["doc_id"], variant))
        if not candidates:
            continue
        if any(
            all(abs(v - variant["per_100g"][n]) <= TOLERANCE for n, v in given.items())
            for _, variant in candidates
        ):
            continue
        doc_id, variant = candidates[0]
        expected = variant["per_100g"]
        off = [n for n, v in given.items() if abs(v - expected[n]) > TOLERANCE]
        found.append(
            {
                "ingredient": ingredient.get("name"),
                "doc_id": doc_id,
                "given": {n: given[n] for n in off},
                "expected": {n: expected[n] for n in off},
            }
        )
    return found
