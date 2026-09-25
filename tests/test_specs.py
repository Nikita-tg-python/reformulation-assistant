"""Structured spec facts (app/specs.py) on the real corpus, and the data check."""

from pathlib import Path

import pytest

from app.agent.tools import AgentTools
from app.specs import NUTRIENTS, parse_allergens, parse_nutrients, spec_mismatches
from tests.fakes import FakePool

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
SPECS = sorted(CORPUS.glob("SPEC-*.md"))


def test_all_twelve_corpus_specs_have_nutrients_and_allergens():
    assert len(SPECS) == 12
    for path in SPECS:
        content = path.read_text()
        variants = parse_nutrients(content)
        assert variants, path.name
        assert all(set(v["per_100g"]) == set(NUTRIENTS) for v in variants), path.name
        assert parse_allergens(content) is not None, path.name


@pytest.mark.parametrize(
    ("doc_id", "allergens", "variants"),
    [
        ("SPEC-001", ["milk"], [None]),
        ("SPEC-002", [], [None]),  # "Кокос не належить до горіхів": not an allergen
        ("SPEC-003", ["gluten"], [None]),
        ("SPEC-007", [], [None, "Чистий екстракт Reb M"]),
        ("SPEC-009", ["milk"], ["Варіант А", "Варіант Б (сухий порошок)"]),
        ("SPEC-011", [], ["Аквафаба", "Лляне яйце (гель)", "KS-2 (суха суміш)"]),
        ("SPEC-012", ["gluten"], ["Пшеничне борошно", "SG-3"]),
    ],
)
def test_spec_variants_and_allergen_codes(doc_id, allergens, variants):
    content = (CORPUS / f"{doc_id}.md").read_text()
    assert parse_allergens(content) == allergens
    assert [v["variant"] for v in parse_nutrients(content)] == variants


def test_first_number_is_taken_and_notes_are_ignored():
    # "- kcal: 20 (0.2–0.4 ккал/г, ...)" -> 20; "sugar_g: 4.7 (уся лактоза)" -> 4.7
    allulose = parse_nutrients((CORPUS / "SPEC-008.md").read_text())[0]["per_100g"]
    milk = parse_nutrients((CORPUS / "SPEC-001.md").read_text())[0]["per_100g"]
    assert allulose["kcal"] == 20
    assert milk == {"kcal": 52, "protein_g": 2.8, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}


def test_other_documents_have_no_sections():
    guide = (CORPUS / "GUIDE-001.md").read_text()
    assert parse_nutrients(guide) is None


def _catalog(*doc_ids: str) -> list[dict]:
    rows = []
    for doc_id in doc_ids:
        content = (CORPUS / f"{doc_id}.md").read_text()
        title = next(ln for ln in content.splitlines() if ln.startswith("title:"))
        rows.append(
            {
                "doc_id": doc_id,
                "title": title.split(":", 1)[1].strip().strip('"'),
                "nutrients": parse_nutrients(content),
            }
        )
    return rows


CATALOG = _catalog("SPEC-001", "SPEC-002", "SPEC-006", "SPEC-011")
MILK = {"kcal": 52, "protein_g": 2.8, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}


def test_faked_nutrients_of_a_known_ingredient_are_reported():
    faked = {**MILK, "protein_g": 3.5, "sugar_g": 1.0}
    found = spec_mismatches([{"name": "молоко 2.5%", "nutrients_per_100g": faked}], CATALOG)
    assert found == [
        {
            "ingredient": "молоко 2.5%",
            "doc_id": "SPEC-001",
            "given": {"protein_g": 3.5, "sugar_g": 1.0},
            "expected": {"protein_g": 2.8, "sugar_g": 4.7},
        }
    ]


@pytest.mark.parametrize(
    ("name", "nutrients"),
    [
        ("молоко 2.5%", {**MILK, "protein_g": 2.84}),  # within 0.05
        ("молоко 2.5%", {"kcal": 52}),  # only given nutrients are compared
        ("молоко", {"kcal": 28, "protein_g": 0.2}),  # also matches SPEC-002: coconut milk
        ("Аквафаба", {"kcal": 18, "protein_g": 1.0}),  # a variant of SPEC-011
        ("soy drink unsweetened", {"kcal": 99}),  # no spec with that name: not checked
        ("молоко 3.2%", {"kcal": 60}),  # a different product than 2.5%
    ],
)
def test_matching_or_unknown_ingredients_are_not_reported(name, nutrients):
    assert spec_mismatches([{"name": name, "nutrients_per_100g": nutrients}], CATALOG) == []


async def test_calc_nutrition_tool_returns_data_mismatches():
    tools = AgentTools(pool=FakePool(specs=CATALOG), embedder=None)
    honest = [{"name": "цукор", "grams": 100, "nutrients_per_100g": {"kcal": 400}}]
    faked = [{"name": "цукор", "grams": 100, "nutrients_per_100g": {"kcal": 250}}]

    ok = await tools.execute("calc_nutrition", {"ingredients": honest})
    bad = await tools.execute("calc_nutrition", {"ingredients": faked})

    assert "data_mismatches" not in ok
    assert bad["per_100g"]["kcal"] == 250  # the numbers are still computed as given
    assert bad["data_mismatches"][0]["doc_id"] == "SPEC-006"
