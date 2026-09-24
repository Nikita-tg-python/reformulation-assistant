"""Agent tools with mocked I/O: Open Food Facts via httpx.MockTransport, DB via FakePool."""

import httpx
import pytest

from app.agent.tools import OFF_SEARCH_URL, OFF_USER_AGENT, TOOL_SPECS, AgentTools
from tests.fakes import FakeEmbedder, FakePool

COCONUT = {
    "code": "123",
    "product_name": "Coconut Milk",
    "nutriments": {
        "energy-kcal_100g": 75,
        "proteins_100g": 0.4,
        "fat_100g": 6.25,
        "carbohydrates_100g": 2.5,
        "sugars_100g": 1.25,
    },
    "allergens_tags": ["en:milk", "en:sesame-seeds", "en:gellan-gum-allergy"],
    "ingredients_text": "coconut extract, water",
}


def off_client(handler) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record)), requests


def hits(*products) -> callable:
    return lambda _request: httpx.Response(200, json={"hits": list(products)})


def tools_with(handler) -> tuple[AgentTools, list[httpx.Request]]:
    client, requests = off_client(handler)
    return AgentTools(FakePool(), FakeEmbedder(), http=client), requests


def test_exactly_three_tools_with_json_schemas():
    assert [t.name for t in TOOL_SPECS] == [
        "search_knowledge_base",
        "lookup_product",
        "calc_nutrition",
    ]
    for spec in TOOL_SPECS:
        assert spec.parameters["type"] == "object"
        assert spec.parameters["required"]


async def test_lookup_product_maps_nutrients_and_eu_allergens():
    tools, requests = tools_with(hits(COCONUT))

    result = await tools.lookup_product("coconut milk")

    assert result["found"] is True
    assert result["source_id"] == "OFF:123"
    assert result["nutrients_per_100g"] == {
        "kcal": 75.0,
        "protein_g": 0.4,
        "fat_g": 6.25,
        "carbs_g": 2.5,
        "sugar_g": 1.25,
    }
    assert result["allergens"] == ["milk", "sesame"]  # non-EU tags are not mapped
    assert result["ingredients_text"] == "coconut extract, water"
    request = requests[0]
    assert str(request.url).startswith(OFF_SEARCH_URL)
    assert request.url.params["q"] == "coconut milk"
    assert request.headers["user-agent"] == OFF_USER_AGENT


async def test_irrelevant_hits_mean_not_found():
    # Open Food Facts search is fuzzy: it returns something for any query.
    tools, _ = tools_with(hits({**COCONUT, "product_name": "123"}))

    result = await tools.lookup_product("zzqxw nonexistent product 123")

    assert result["found"] is False
    assert "matches" in result["message"]


async def test_prefers_relevant_hit_with_nutrition_data():
    empty = {**COCONUT, "code": "1", "nutriments": {}}
    french = {**COCONUT, "code": "2", "product_name": "Lait de coco"}
    tools, _ = tools_with(hits(french, empty, COCONUT))

    result = await tools.lookup_product("coconut milk")

    assert result["source_id"] == "OFF:123"


async def test_no_hits_means_not_found():
    tools, _ = tools_with(hits())
    assert (await tools.lookup_product("anything"))["found"] is False


async def test_results_are_cached_per_run_ignoring_case_and_spaces():
    tools, requests = tools_with(hits(COCONUT))

    first = await tools.lookup_product("coconut milk")
    second = await tools.lookup_product("  Coconut   MILK ")

    assert first == second
    assert len(requests) == 1
    # A new run (new AgentTools) has its own cache.
    fresh, fresh_requests = tools_with(hits(COCONUT))
    await fresh.lookup_product("coconut milk")
    assert len(fresh_requests) == 1


def _raise(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (_raise(httpx.ReadTimeout("slow")), "timeout"),
        (_raise(httpx.ConnectError("dns")), "network_error"),
        (lambda _r: httpx.Response(500), "http_error"),
        (lambda _r: httpx.Response(200, text="not json"), "network_error"),
    ],
    ids=["timeout", "connect", "http-500", "bad-json"],
)
async def test_network_errors_are_returned_as_data(handler, code):
    tools, requests = tools_with(handler)

    result = await tools.lookup_product("coconut milk")
    assert result == {"error": {"code": code, "message": result["error"]["message"]}}

    # Errors are not cached: the next call tries again.
    await tools.lookup_product("coconut milk")
    assert len(requests) == 2


async def test_search_knowledge_base_returns_chunks():
    chunk = {"doc_id": "SPEC-011", "title": "Замінники яйця", "text": "Аквафаба…", "score": 0.9}
    tools = AgentTools(FakePool(chunks=[chunk]), FakeEmbedder())

    result = await tools.execute("search_knowledge_base", {"query": "аквафаба", "top_k": 3})

    assert result == {"results": [chunk]}


async def test_execute_rejects_unknown_tools_and_bad_arguments():
    tools = AgentTools(FakePool(), FakeEmbedder())

    assert (await tools.execute("drop_tables", {}))["error"]["code"] == "unknown_tool"
    bad = await tools.execute("search_knowledge_base", {"query": "", "top_k": 99})
    assert bad["error"]["code"] == "invalid_arguments"
    assert "query" in bad["error"]["message"]
    assert "top_k" in bad["error"]["message"]
    assert (await tools.execute("lookup_product", {}))["error"]["code"] == "invalid_arguments"
