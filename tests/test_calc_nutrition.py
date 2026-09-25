import pytest
from pydantic import ValidationError

from app.agent.tools import NUTRIENTS, AgentTools, NutritionIngredient, calc_nutrition
from tests.fakes import FakePool


def ing(name: str, grams: float, **nutrients: float | None) -> NutritionIngredient:
    return NutritionIngredient(name=name, grams=grams, nutrients_per_100g=nutrients)


MILK = {"kcal": 52, "protein_g": 2.8, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
SUGAR = {"kcal": 400, "protein_g": 0, "fat_g": 0, "carbs_g": 100, "sugar_g": 100}
STRAWBERRY = {"kcal": 32, "protein_g": 0.7, "fat_g": 0.3, "carbs_g": 7.7, "sugar_g": 4.9}
STARTER = {"kcal": 58, "protein_g": 3.1, "fat_g": 2.4, "carbs_g": 5.2, "sugar_g": 3.9}


def yogurt(scale: float = 1.0) -> list[NutritionIngredient]:
    return [
        ing("молоко 2.5%", 800 * scale, **MILK),
        ing("цукор", 90 * scale, **SUGAR),
        ing("полуниця заморожена", 100 * scale, **STRAWBERRY),
        ing("закваска", 10 * scale, **STARTER),
    ]


def test_weighted_average_matches_trial_001_control():
    # Same recipe and numbers as the control in data/corpus/TRIAL-001.md.
    result = calc_nutrition(yogurt())
    assert result["total_grams"] == 1000
    assert result["per_100g"] == {
        "kcal": 81.4,
        "protein_g": 2.3,
        "fat_g": 2.1,
        "carbs_g": 13.6,
        "sugar_g": 13.3,
    }
    assert result["missing"] == []
    assert result["warnings"] == []


def test_single_ingredient_returns_its_own_values():
    assert calc_nutrition([ing("молоко", 250, **MILK)])["per_100g"] == MILK


def test_result_does_not_depend_on_batch_size_or_order():
    base = calc_nutrition(yogurt())["per_100g"]
    assert calc_nutrition(yogurt(scale=0.37))["per_100g"] == base
    assert calc_nutrition(list(reversed(yogurt())))["per_100g"] == base


def test_zero_gram_ingredient_does_not_change_result():
    with_zero = [*yogurt(), ing("ароматизатор", 0, kcal=900)]
    assert calc_nutrition(with_zero)["per_100g"] == calc_nutrition(yogurt())["per_100g"]


@pytest.mark.parametrize(
    "ingredients",
    [[], [ing("цукор", 0, **SUGAR)], [ing("вода", 0), ing("сіль", 0)]],
    ids=["empty", "one-zero", "all-zero"],
)
def test_zero_total_mass_gives_nulls_and_warning(ingredients):
    result = calc_nutrition(ingredients)
    assert result["total_grams"] == 0
    assert result["per_100g"] == dict.fromkeys(NUTRIENTS)
    assert any("total mass is 0" in w for w in result["warnings"])


def test_missing_nutrients_count_as_zero_and_are_reported():
    result = calc_nutrition(
        [
            ing("молоко", 500, **MILK),
            ing("вода", 500),  # no nutrients at all
            ing("цукор", 0),  # zero grams: irrelevant, not reported
        ]
    )
    assert result["per_100g"] == {k: round(v / 2, 1) for k, v in MILK.items()}
    assert result["missing"] == [{"ingredient": "вода", "nutrients": list(NUTRIENTS)}]
    assert any("lower bounds" in w for w in result["warnings"])


def test_null_and_partial_nutrients():
    result = calc_nutrition([ing("білок", 100, kcal=370, protein_g=80, sugar_g=None)])
    assert result["per_100g"] == {
        "kcal": 370,
        "protein_g": 80,
        "fat_g": 0,
        "carbs_g": 0,
        "sugar_g": 0,
    }
    assert result["missing"] == [
        {"ingredient": "білок", "nutrients": ["fat_g", "carbs_g", "sugar_g"]}
    ]


def test_unknown_nutrient_keys_are_ignored():
    result = calc_nutrition([ing("яблуко", 100, kcal=52, fiber_g=2.4)])
    assert result["per_100g"]["kcal"] == 52
    assert "fiber_g" not in result["per_100g"]


def test_negative_grams_rejected_by_model():
    with pytest.raises(ValidationError):
        ing("молоко", -1, **MILK)


async def test_tool_call_with_invalid_arguments_returns_error_not_exception():
    tools = AgentTools(pool=FakePool(), embedder=None)  # the pool only serves the spec check
    result = await tools.execute(
        "calc_nutrition", {"ingredients": [{"name": "молоко", "grams": -5}]}
    )
    assert result["error"]["code"] == "invalid_arguments"
    assert "grams" in result["error"]["message"]


async def test_tool_call_happy_path_through_execute():
    tools = AgentTools(pool=FakePool(), embedder=None)
    result = await tools.execute(
        "calc_nutrition",
        {"ingredients": [i.model_dump() for i in yogurt()]},
    )
    assert result["per_100g"]["kcal"] == 81.4


def test_result_lies_between_ingredient_extremes():
    # A weighted average can never leave the range of its inputs.
    result = calc_nutrition(yogurt())["per_100g"]
    for nutrient in NUTRIENTS:
        values = [i.nutrients_per_100g[nutrient] for i in yogurt()]
        assert min(values) <= result[nutrient] <= max(values)


def test_fractional_grams_and_rounding():
    result = calc_nutrition([ing("a", 0.3, kcal=375), ing("b", 99.7, kcal=0)])
    assert result["total_grams"] == 100.0
    assert result["per_100g"]["kcal"] == 1.1  # 1.125 rounded to one decimal
