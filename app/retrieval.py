"""Vector search over chunks in pgvector (cosine distance, hnsw index)."""

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
