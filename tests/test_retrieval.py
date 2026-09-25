"""Reciprocal Rank Fusion and search mode selection, without a database."""

import pytest

from app.retrieval import RRF_K, rrf, search_chunks
from tests.fakes import FakeEmbedder, FakePool


def test_rrf_sums_reciprocal_ranks_across_rankings():
    # b: 1/(60+2) + 1/(60+1) beats a: 1/(60+1) + 1/(60+3), c appears only once.
    assert rrf([["a", "b", "c"], ["b", "d", "a"]]) == ["b", "a", "d", "c"]


def test_rrf_item_in_both_rankings_beats_a_single_first_place():
    assert rrf([["x", "y"], ["y"]]) == ["y", "x"]
    assert 1 / (RRF_K + 2) + 1 / (RRF_K + 1) > 1 / (RRF_K + 1)


def test_rrf_ties_keep_the_order_of_the_first_ranking():
    assert rrf([["a", "b"], ["b", "a"]]) == ["a", "b"]  # equal sums
    assert rrf([["v1", "v2"], []]) == ["v1", "v2"]  # empty full-text result: vector order


def test_rrf_edge_cases_empty_input_and_k_zero():
    assert rrf([["a", "b"], ["b", "a"]], k=0) == ["a", "b"]
    assert rrf([]) == []


CHUNKS = [
    {"doc_id": "SPEC-002", "title": "Кокос", "text": "кокосове молоко", "score": 0.9},
    {"doc_id": "SPEC-011", "title": "Яйце", "text": "аквафаба", "score": 0.8},
]


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
async def test_both_modes_return_top_k_with_cosine_scores(mode):
    chunks = await search_chunks(FakePool(chunks=CHUNKS), FakeEmbedder(), "молоко", 1, mode=mode)
    assert [(c.doc_id, c.score) for c in chunks] == [("SPEC-002", 0.9)]


async def test_mode_defaults_to_the_search_mode_setting(monkeypatch):
    from app import retrieval
    from app.config import Settings

    seen = []

    class Pool(FakePool):
        async def fetch(self, query, *args):
            seen.append("fulltext" if "websearch_to_tsquery" in query else "vector")
            return await super().fetch(query, *args)

    for mode, expected in [("vector", ["vector"]), ("hybrid", ["vector", "fulltext"])]:
        seen.clear()
        monkeypatch.setattr(retrieval, "get_settings", lambda m=mode: Settings(search_mode=m))
        await search_chunks(Pool(chunks=CHUNKS), FakeEmbedder(), "молоко", 2)
        assert seen == expected
