"""Chunk search: vector (pgvector, cosine, hnsw) or hybrid (vector + full text, RRF)."""

import re
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import asyncpg

from app.config import SearchMode, get_settings
from app.embeddings import Embedder

RRF_K = 60  # the constant from the RRF paper (Cormack et al., 2009)
CANDIDATES = 20  # per ranking in hybrid mode, before fusion


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float  # cosine similarity, 1 - cosine distance; higher is closer


async def has_chunks(pool: asyncpg.Pool) -> bool:
    return await pool.fetchval("SELECT EXISTS (SELECT 1 FROM chunks)")


async def search_chunks(
    pool: asyncpg.Pool,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
    mode: SearchMode | None = None,
) -> list[RetrievedChunk]:
    """top_k chunks for the query; mode defaults to SEARCH_MODE.

    vector: nearest chunks by cosine distance.
    hybrid: vector top 20 and full-text top 20, fused with Reciprocal Rank Fusion. Full text
    catches what embeddings miss: document codes (SPEC-001), E-numbers, exact terms.
    `score` stays the cosine similarity in both modes (the order in hybrid comes from RRF).
    """
    mode = mode or get_settings().search_mode
    query_vec = await embedder.embed_query(query)
    if mode == "vector":
        return [_chunk(r) for r in await _vector(pool, query_vec, top_k)]

    limit = max(CANDIDATES, top_k)
    by_vector = await _vector(pool, query_vec, limit)
    by_text = await pool.fetch(
        # The ORDER BY ts_rank_cd is computed over GIN matches only; weight A (doc_id) ranks
        # a document's own chunks above chunks that merely cite its code.
        """
        SELECT c.id, c.doc_id, d.title, c.text, 1 - (c.embedding <=> $2) AS score
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id,
               websearch_to_tsquery('simple', $1) AS q
         WHERE c.tsv @@ q
         ORDER BY ts_rank_cd(c.tsv, q) DESC, c.id
         LIMIT $3
        """,
        query,
        query_vec,
        limit,
    )
    rows = {r["id"]: r for r in [*by_vector, *by_text]}
    fused = rrf([[r["id"] for r in by_vector], [r["id"] for r in by_text]])
    return [_chunk(rows[i]) for i in fused[:top_k]]


async def _vector(pool: asyncpg.Pool, query_vec, limit: int) -> list:
    # `ORDER BY embedding <=> $1 LIMIT k` is the exact shape the hnsw index
    # (vector_cosine_ops) can serve; the join to documents happens after.
    return await pool.fetch(
        """
        SELECT c.id, c.doc_id, d.title, c.text, 1 - (c.embedding <=> $1) AS score
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
         ORDER BY c.embedding <=> $1
         LIMIT $2
        """,
        query_vec,
        limit,
    )


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Reciprocal Rank Fusion: score(item) = sum over rankings of 1 / (k + rank), rank from 1.

    Only ranks are used, so cosine similarity and ts_rank need no common scale. Ties keep
    first-seen order, i.e. the order of the first ranking (vector in hybrid search).
    """
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


def _chunk(r) -> RetrievedChunk:
    return RetrievedChunk(r["doc_id"], r["title"], r["text"], round(float(r["score"]), 4))


_NUTRIENTS_SECTION = re.compile(r"^## Нутрієнти на 100 г[^\n]*\n(.*?)(?=^## |\Z)", re.M | re.S)
NUTRIENTS_TEXT_LIMIT = 300


async def spec_nutrients(pool: asyncpg.Pool, doc_ids: list[str]) -> dict[str, str]:
    """The "Нутрієнти на 100 г" section of ingredient specs, as one short line per doc.

    The numbers sit near the end of a spec, often outside the chunk that search returns,
    so the agent gets them per document instead of hunting for them.
    """
    rows = await pool.fetch(
        "SELECT doc_id, content FROM documents "
        "WHERE doc_id = ANY($1::text[]) AND doc_type = 'ingredient_spec'",
        doc_ids,
    )
    sections = {}
    for row in rows:
        match = _NUTRIENTS_SECTION.search(row["content"])
        if match:
            lines = (
                " ".join(ln.strip().removeprefix("- ").split())
                for ln in match.group(1).splitlines()
            )
            sections[row["doc_id"]] = "; ".join(ln for ln in lines if ln)[:NUTRIENTS_TEXT_LIMIT]
    return sections
