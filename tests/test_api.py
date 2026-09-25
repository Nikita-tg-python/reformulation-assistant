"""HTTP API through httpx.AsyncClient + ASGITransport, with FakeLLM and FakeEmbedder.

Unit tests use FakePool and need nothing. Integration tests need Postgres with pgvector
(DATABASE_URL, e.g. the docker compose db) and run in a throwaway database that is
created and dropped per test, so real data is never touched.
"""

import json
import os
import uuid
from urllib.parse import urlparse, urlunparse

import asyncpg
import httpx
import pytest

from app.db import create_pool
from app.llm.base import LLMNotConfiguredError, ToolCall
from app.llm.fake import FakeLLM
from app.main import app
from app.routers.ask import NO_DATA_ANSWER
from tests.fakes import FakeEmbedder, FakePool

EGG_CHUNK = {
    "doc_id": "SPEC-011",
    "title": "Замінники яйця",
    "text": "Аквафаба 45 г замінює одне яйце в бісквіті.",
    "score": 0.91,
}
MILK_CHUNK = {"doc_id": "SPEC-002", "title": "Кокосове молоко", "text": "…", "score": 0.52}


def no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected outbound request {request.url}")


@pytest.fixture
async def make_client():
    """Point the app at the given pool and LLM, return an HTTP client for it."""
    clients = []

    async def factory(pool, llm=None) -> httpx.AsyncClient:
        app.state.pool = pool
        app.state.embedder = FakeEmbedder()
        app.state.llm = llm or FakeLLM()
        app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(no_network))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        clients.append(client)
        return client

    yield factory
    for client in clients:
        await client.aclose()


def answer(found: bool, text: str = "", doc_ids=()) -> str:
    return json.dumps({"found": found, "answer": text, "doc_ids": list(doc_ids)})


# ---------- unit: no database ----------


async def test_health_ok_and_request_id(make_client):
    client = await make_client(FakePool())

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "llm_provider": "groq"}
    assert len(response.headers["x-request-id"]) == 16

    echoed = await client.get("/health", headers={"X-Request-ID": "trace-me-1"})
    assert echoed.headers["x-request-id"] == "trace-me-1"
    unsafe = await client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})
    assert unsafe.headers["x-request-id"] != "bad id\nwith newline"


async def test_health_degraded_when_db_is_down(make_client):
    client = await make_client(FakePool(healthy=False))

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["db"] == "error"


async def test_errors_use_the_common_format(make_client):
    client = await make_client(FakePool())

    not_found = await client.get("/nope")
    invalid = await client.post(
        "/documents", json={"doc_id": "X", "title": "t", "doc_type": "recipe", "content": "c"}
    )
    bad_goal = await client.post(
        "/reformulate",
        json={
            "product_name": "x",
            "ingredients": [{"name": "a", "grams": 1}],
            "goal": "reduce_sugar",
            "goal_params": {"percent": 60},
        },
    )

    assert not_found.status_code == 404
    assert not_found.json() == {"error": {"code": "not_found", "message": "Not Found"}}
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert "doc_type" in invalid.json()["error"]["message"]
    assert bad_goal.status_code == 422
    assert "percent" in bad_goal.json()["error"]["message"]


async def test_ask_on_empty_knowledge_base_is_409(make_client):
    client = await make_client(FakePool(chunks=[]))

    response = await client.post("/ask", json={"question": "Чим замінити яйце?"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "empty_knowledge_base"


async def test_ask_answers_only_from_excerpts_and_returns_cited_sources(make_client):
    def reply(messages):
        prompt = messages[-1].content
        assert messages[0].role == "system"
        assert "ONLY" in messages[0].content
        assert "[SPEC-011]" in prompt
        assert "Аквафаба 45 г" in prompt
        return answer(True, "Аквафаба, 45 г на яйце (SPEC-011).", ["SPEC-011"])

    client = await make_client(FakePool(chunks=[EGG_CHUNK, MILK_CHUNK]), FakeLLM([reply]))

    response = await client.post("/ask", json={"question": "Чим замінити яйце?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Аквафаба, 45 г на яйце (SPEC-011)."
    assert body["sources"] == [
        {
            "doc_id": "SPEC-011",
            "title": "Замінники яйця",
            "chunk_text": EGG_CHUNK["text"],
            "score": 0.91,
        }
    ]


async def test_ask_refuses_when_excerpts_have_no_answer(make_client):
    llm = FakeLLM([answer(False)])
    client = await make_client(FakePool(chunks=[MILK_CHUNK]), llm)

    response = await client.post("/ask", json={"question": "Яка погода в Києві?"})

    assert response.json() == {"answer": NO_DATA_ANSWER, "sources": []}


async def test_ask_invalid_llm_json_is_502(make_client):
    client = await make_client(FakePool(chunks=[EGG_CHUNK]), FakeLLM(["not json"]))

    response = await client.post("/ask", json={"question": "x"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_bad_response"


async def test_ask_without_llm_key_is_503(make_client):
    def not_configured(_messages):
        raise LLMNotConfiguredError("GEMINI_API_KEY is not set")

    client = await make_client(FakePool(chunks=[EGG_CHUNK]), FakeLLM([not_configured]))

    response = await client.post("/ask", json={"question": "x"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


# ---------- integration: real Postgres + pgvector ----------

DATABASE_URL = os.environ.get("DATABASE_URL")
needs_db = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")


@pytest.fixture
async def db_pool():
    """A fresh database with migrations applied, dropped after the test."""
    name = f"pytest_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(DATABASE_URL)
    await admin.execute(f'CREATE DATABASE "{name}"')
    pool = await create_pool(urlunparse(urlparse(DATABASE_URL)._replace(path=f"/{name}")))
    try:
        yield pool
    finally:
        await pool.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


def document(doc_id: str, content: str, title: str = "Документ") -> dict:
    return {"doc_id": doc_id, "title": title, "doc_type": "ingredient_spec", "content": content}


EGG_DOC = document(
    "SPEC-011",
    "Аквафаба замінює яйце в бісквіті: 45 г аквафаби на одне яйце.",
    "Замінники яйця",
)
MILK_DOC = document("SPEC-002", "Кокосове молоко 2.5% жиру для ферментованого йогурту.")


@needs_db
async def test_documents_ingest_and_reingest_without_duplicates(make_client, db_pool):
    client = await make_client(db_pool)
    long_text = " ".join(f"слово{i}" for i in range(900))  # 900 tokens -> 3 chunks

    first = await client.post("/documents", json=document("DOC-1", long_text))
    again = await client.post("/documents", json=document("DOC-1", long_text))
    shorter = await client.post("/documents", json=document("DOC-1", "короткий текст", "Новий"))

    assert first.json() == {"doc_id": "DOC-1", "chunks_created": 3}
    assert again.json()["chunks_created"] == 3
    assert shorter.json()["chunks_created"] == 1
    assert await db_pool.fetchval("SELECT count(*) FROM chunks WHERE doc_id = 'DOC-1'") == 1
    assert await db_pool.fetchval("SELECT count(*) FROM documents") == 1
    assert await db_pool.fetchval("SELECT title FROM documents WHERE doc_id = 'DOC-1'") == "Новий"


@needs_db
async def test_ask_end_to_end_with_vector_search(make_client, db_pool):
    def reply(messages):
        prompt = messages[-1].content
        first_excerpt = prompt.split("[", 1)[1]
        assert first_excerpt.startswith("SPEC-011]"), "closest chunk should come first"
        return answer(True, "45 г аквафаби на яйце (SPEC-011).", ["SPEC-011"])

    client = await make_client(db_pool, FakeLLM([reply]))

    empty = await client.post("/ask", json={"question": "Чим замінити яйце в бісквіті?"})
    for doc in (EGG_DOC, MILK_DOC):
        assert (await client.post("/documents", json=doc)).status_code == 200
    response = await client.post("/ask", json={"question": "Чим замінити яйце в бісквіті?"})

    assert empty.status_code == 409
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert [s["doc_id"] for s in sources] == ["SPEC-011"]
    assert 0 < sources[0]["score"] <= 1


@needs_db
async def test_hybrid_search_puts_the_document_with_that_code_first(make_client, db_pool):
    from app.retrieval import search_chunks

    # A chunk never contains its own document code: only the full-text index knows it
    # (chunks.tsv includes doc_id). Vector search has nothing to match "SPEC-001" against.
    client = await make_client(db_pool)
    docs = [
        document("SPEC-001", "Молоко коров'яче 2.5% жиру: білок 2.8 г, лактоза 4.7 г."),
        document("SPEC-010", "Яйце куряче, меланж пастеризований для бісквіта."),
        EGG_DOC,
        MILK_DOC,
    ]
    for doc in docs:
        assert (await client.post("/documents", json=doc)).status_code == 200

    hybrid = await search_chunks(db_pool, FakeEmbedder(), "SPEC-001", top_k=3, mode="hybrid")
    vector = await search_chunks(db_pool, FakeEmbedder(), "SPEC-001", top_k=3, mode="vector")

    assert hybrid[0].doc_id == "SPEC-001"
    assert 0 <= hybrid[0].score <= 1  # still cosine similarity, not the RRF score
    assert len(vector) == 3  # vector mode ignores the full-text index


@needs_db
async def test_ingest_fills_structured_spec_facts_and_calc_checks_them(make_client, db_pool):
    from pathlib import Path

    from app.agent.tools import AgentTools
    from app.ingest import load_folder

    client = await make_client(db_pool)
    corpus = {
        doc.doc_id: doc for _, doc in load_folder(Path(__file__).parent.parent / "data/corpus")
    }
    for doc_id in ("SPEC-001", "SPEC-012", "GUIDE-001"):
        body = corpus[doc_id].model_dump()
        assert (await client.post("/documents", json=body)).status_code == 200

    rows = {
        r["doc_id"]: r
        for r in await db_pool.fetch("SELECT doc_id, nutrients, allergens FROM documents")
    }
    assert rows["SPEC-001"]["allergens"] == ["milk"]
    assert json.loads(rows["SPEC-012"]["nutrients"])[1]["variant"] == "SG-3"
    assert rows["GUIDE-001"]["nutrients"] is None
    assert rows["GUIDE-001"]["allergens"] is None

    tools = AgentTools(db_pool, FakeEmbedder())
    found = await tools.execute("search_knowledge_base", {"query": "молоко коров'яче"})
    milk = next(r for r in found["results"] if r["doc_id"] == "SPEC-001")
    assert milk["allergens"] == ["milk"]
    assert "kcal: 52" in milk["nutrients_per_100g"]

    faked = {"kcal": 52, "protein_g": 3.5, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
    calc = await tools.execute(
        "calc_nutrition",
        {"ingredients": [{"name": "молоко 2.5%", "grams": 800, "nutrients_per_100g": faked}]},
    )
    assert calc["data_mismatches"][0]["given"] == {"protein_g": 3.5}


@needs_db
async def test_reformulate_records_successful_and_timed_out_runs(make_client, db_pool, monkeypatch):
    from app.config import Settings
    from app.routers import reformulate as reformulate_router

    # This test drives the free tool-calling loop, whatever AGENT_MODE defaults to.
    monkeypatch.setattr(reformulate_router, "get_settings", lambda: Settings(agent_mode="loop"))
    recipe = [{"name": "молоко", "grams": 800, "nutrients_per_100g": {"kcal": 52}}]
    coconut = [{"name": "кокосове молоко", "grams": 800, "nutrients_per_100g": {"kcal": 28}}]
    final = {
        "substitutions": [
            {
                "original": "молоко",
                "replacement": "кокосове молоко",
                "grams": 800,
                "rationale": "SPEC-002",
                "sources": ["SPEC-002"],
            }
        ],
        "allergens_before": ["milk"],
        "allergens_after": [],
        "nutrition_per_100g": {"before": {"kcal": 52}, "after": {"kcal": 28}},
    }
    script = [
        [ToolCall("1", "search_knowledge_base", {"query": "кокосове молоко йогурт"})],
        [ToolCall("2", "calc_nutrition", {"ingredients": recipe})],
        [ToolCall("3", "calc_nutrition", {"ingredients": coconut})],
        json.dumps(final, ensure_ascii=False),
    ]
    looping = [
        [ToolCall(str(i), "search_knowledge_base", {"query": f"варіант {i}"})] for i in range(10)
    ]
    request = {
        "product_name": "Йогурт",
        "ingredients": [{"name": "молоко", "grams": 800}],
        "goal": "remove_allergen",
        "goal_params": {"allergen": "milk"},
    }
    client = await make_client(db_pool, FakeLLM(script))
    await client.post("/documents", json=MILK_DOC)

    ok = await client.post("/reformulate", json=request, headers={"X-Request-ID": "run-ok"})
    app.state.llm = FakeLLM(looping)
    timeout = await client.post("/reformulate", json=request, headers={"X-Request-ID": "run-loop"})

    assert ok.status_code == 200
    body = ok.json()
    assert body["substitutions"][0]["sources"] == ["SPEC-002"]
    assert [e.get("tool") for e in body["trace"]].count("calc_nutrition") == 2
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "agent_timeout"
    assert len(timeout.json()["error"]["trace"]) == 7  # 1 facts search + 6 iterations

    rows = await db_pool.fetch(
        "SELECT status, request_id, duration_ms, jsonb_array_length(trace) AS steps, "
        "request->>'goal' AS goal FROM reformulation_runs ORDER BY id"
    )
    assert [(r["status"], r["request_id"], r["steps"], r["goal"]) for r in rows] == [
        ("ok", "run-ok", 5, "remove_allergen"),  # 1 facts search + 3 tool calls + final
        ("agent_timeout", "run-loop", 7, "remove_allergen"),
    ]
    assert all(r["duration_ms"] is not None for r in rows)


async def test_reformulate_uses_the_pipeline_when_agent_mode_is_pipeline(make_client, monkeypatch):
    from app.agent.tools import CalcNutritionArgs, calc_nutrition
    from app.config import Settings
    from app.routers import reformulate as reformulate_router

    monkeypatch.setattr(reformulate_router, "get_settings", lambda: Settings(agent_mode="pipeline"))
    milk = {"kcal": 52, "protein_g": 2.8, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
    soy = {"kcal": 34, "protein_g": 3.3, "fat_g": 1.9, "carbs_g": 0.6, "sugar_g": 0.3}
    spec = (
        "## Нутрієнти на 100 г\n\n- kcal: {kcal}\n- protein_g: {protein_g}\n- fat_g: {fat_g}\n"
        "- carbs_g: {carbs_g}\n- sugar_g: {sugar_g}\n"
    )
    pool = FakePool(
        chunks=[
            {
                "doc_id": "SPEC-004",
                "title": "Соєвий напій",
                "text": "Соя для йогурту.",
                "score": 0.9,
            },
            {"doc_id": "SPEC-001", "title": "Молоко", "text": "Молоко 2.5%.", "score": 0.8},
        ],
        documents={"SPEC-004": spec.format(**soy), "SPEC-001": spec.format(**milk)},
    )

    def per_100g(nutrients):
        args = CalcNutritionArgs(
            ingredients=[{"name": "x", "grams": 800, "nutrients_per_100g": nutrients}]
        )
        return calc_nutrition(args.ingredients)["per_100g"]

    choice = {
        "substitutions": [
            {
                "original": "молоко",
                "replacement": "соєвий напій",
                "grams": 800,
                "sources": ["SPEC-004"],
            }
        ],
        "nutrients": {
            "молоко": {"per_100g": milk, "source": "SPEC-001"},
            "соєвий напій": {"per_100g": soy, "source": "SPEC-004"},
        },
    }
    answer = {
        "substitutions": [
            {
                "original": "молоко",
                "replacement": "соєвий напій",
                "grams": 800,
                "rationale": "SPEC-004",
                "sources": ["SPEC-004"],
            }
        ],
        "allergens_before": ["milk"],
        "allergens_after": ["soybeans"],
        "nutrition_per_100g": {"before": per_100g(milk), "after": per_100g(soy)},
        "warnings": ["Новий алерген: соя."],
    }
    llm = FakeLLM([json.dumps(choice, ensure_ascii=False), json.dumps(answer, ensure_ascii=False)])
    client = await make_client(pool, llm)

    response = await client.post(
        "/reformulate",
        json={
            "product_name": "Йогурт",
            "ingredients": [{"name": "молоко", "grams": 800}],
            "goal": "remove_allergen",
            "goal_params": {"allergen": "milk"},
        },
    )

    assert response.status_code == 200, response.text
    assert len(llm.calls) == 2
    trace = response.json()["trace"]
    assert [e["purpose"] for e in trace if e["type"] == "llm_call"] == ["choose", "final"]
    assert response.json()["allergens_after"] == ["soybeans"]
