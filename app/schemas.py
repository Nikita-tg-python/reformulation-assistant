from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocType = Literal["ingredient_spec", "trial_report", "guideline"]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class AgentErrorBody(ErrorBody):
    trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentErrorResponse(BaseModel):
    error: AgentErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: Literal["ok", "error"]
    llm_provider: str


class DocumentIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "doc_id": "SPEC-099",
                    "title": "Гороховий білок ізолят 80%",
                    "doc_type": "ingredient_spec",
                    "content": "## Функція в продукті\nПідвищує білок у рослинних йогуртах...",
                }
            ]
        },
    )

    doc_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1, max_length=300)
    doc_type: DocType
    content: str = Field(min_length=1, max_length=200_000)


class DocumentIngestResponse(BaseModel):
    doc_id: str
    chunks_created: int


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{"question": "Чим замінити яйце в бісквіті?", "top_k": 5}]},
    )

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AskSource(BaseModel):
    doc_id: str
    title: str
    chunk_text: str
    score: float = Field(description="Cosine similarity to the question, 1 - cosine distance")


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]


# ---------- /reformulate ----------

# EU 1169/2011 Annex II, codes as in data/corpus/GUIDE-001.md.
EUAllergen = Literal[
    "gluten", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk",
    "nuts", "celery", "mustard", "sesame", "sulphites", "lupin", "molluscs",
]  # fmt: skip
Goal = Literal["remove_allergen", "reduce_sugar", "make_vegan"]


class RemoveAllergenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allergen: EUAllergen


class ReduceSugarParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    percent: int = Field(ge=10, le=50)


class MakeVeganParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


_GOAL_PARAMS: dict[str, type[BaseModel]] = {
    "remove_allergen": RemoveAllergenParams,
    "reduce_sugar": ReduceSugarParams,
    "make_vegan": MakeVeganParams,
}


class RecipeIngredient(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    grams: float = Field(gt=0, le=1_000_000)


class ReformulateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "product_name": "Полуничний йогурт 2.5%",
                    "ingredients": [
                        {"name": "молоко 2.5%", "grams": 800},
                        {"name": "цукор", "grams": 90},
                        {"name": "полуниця заморожена", "grams": 100},
                        {"name": "закваска", "grams": 10},
                    ],
                    "goal": "remove_allergen",
                    "goal_params": {"allergen": "milk"},
                }
            ]
        },
    )

    product_name: str = Field(min_length=1, max_length=200)
    ingredients: list[RecipeIngredient] = Field(min_length=1, max_length=50)
    goal: Goal
    goal_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_goal_params(self) -> Self:
        # Validate goal_params against the goal's own model; errors surface as 422.
        self.goal_params = _GOAL_PARAMS[self.goal].model_validate(self.goal_params).model_dump()
        return self


class NutritionValues(BaseModel):
    kcal: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carbs_g: float | None = None
    sugar_g: float | None = None


class NutritionComparison(BaseModel):
    before: NutritionValues
    after: NutritionValues


class Substitution(BaseModel):
    original: str = Field(min_length=1)
    replacement: str = Field(min_length=1)
    grams: float = Field(ge=0)
    rationale: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"

    @model_validator(mode="after")
    def _sources_or_low_confidence(self) -> Self:
        if not self.sources and self.confidence != "low":
            raise ValueError(
                f"substitution {self.original!r} -> {self.replacement!r} has no sources: "
                "cite a doc_id or OFF product, or set confidence to 'low'"
            )
        return self


class ReformulationDraft(BaseModel):
    """What the LLM must return. The API response adds the trace."""

    substitutions: list[Substitution]
    allergens_before: list[EUAllergen]
    allergens_after: list[EUAllergen]
    nutrition_per_100g: NutritionComparison
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _warn_low_confidence(self) -> Self:
        # Spec: a low-confidence substitution always shows up in warnings.
        for s in self.substitutions:
            if s.confidence == "low" and not any(s.replacement in w for w in self.warnings):
                self.warnings.append(
                    f"Заміна «{s.original}» → «{s.replacement}» має низьку впевненість: "
                    "немає підтвердження в базі знань чи Open Food Facts."
                )
        return self


class ReformulateResponse(ReformulationDraft):
    trace: list[dict[str, Any]]
