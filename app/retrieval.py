"""Vector search over chunks in pgvector (cosine distance, hnsw index)."""

import re
from dataclasses import dataclass

import asyncpg

from app.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float  # cosine similarity, 1 - cosine distance; higher is closer


async def has_chunks(pool: asyncpg.Pool) -> bool:
    return await pool.fetchval("SELECT EXISTS (SELECT 1 FROM chunks)")


async def search_chunks(
    pool: asyncpg.Pool, embedder: Embedder, query: str, top_k: int = 5
) -> list[RetrievedChunk]:
    """top_k chunks closest to the query. `ORDER BY embedding <=> $1 LIMIT k` is the exact
    shape the hnsw index (vector_cosine_ops) can serve; the join to documents happens after."""
    query_vec = await embedder.embed_query(query)
    rows = await pool.fetch(
        """
        SELECT c.doc_id, d.title, c.text, 1 - (c.embedding <=> $1) AS score
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
         ORDER BY c.embedding <=> $1
         LIMIT $2
        """,
        query_vec,
        top_k,
    )
    return [
        RetrievedChunk(r["doc_id"], r["title"], r["text"], round(float(r["score"]), 4))
        for r in rows
    ]


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
