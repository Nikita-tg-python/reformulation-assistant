"""RAG retrieval quality: recall@5 and MRR@5 for vector and hybrid search.

    python -m eval.run            # needs DATABASE_URL (a Postgres with pgvector) and the model

Creates a temporary database next to DATABASE_URL, ingests data/corpus/ into it, asks every
question from eval/questions.jsonl in both search modes and drops the database. Metrics are
per document: chunks are deduplicated by doc_id before taking the top 5. A question counts
as found if any of its expected documents is there. Exit code 1 if recall@5 < 0.8 in any mode.
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import get_args
from urllib.parse import urlparse, urlunparse

import asyncpg

from app.config import SearchMode, get_settings
from app.db import create_pool
from app.embeddings import OnnxEmbedder
from app.ingest import ingest_document, load_folder
from app.retrieval import search_chunks

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.jsonl"
CORPUS = ROOT / "data" / "corpus"
K = 5
CHUNKS_PER_QUERY = 20  # enough chunks to fill the top 5 distinct documents
MIN_RECALL = 0.8


def top_documents(doc_ids: list[str], k: int = K) -> list[str]:
    return list(dict.fromkeys(doc_ids))[:k]


def recall_and_rr(ranked: list[str], expected: set[str]) -> tuple[bool, float]:
    """Found in the ranked documents, and 1/rank of the first expected one (0 if absent)."""
    rank = next((n for n, d in enumerate(ranked, start=1) if d in expected), None)
    return rank is not None, 1 / rank if rank else 0.0


async def evaluate(pool: asyncpg.Pool, embedder: OnnxEmbedder, questions: list[dict]) -> dict:
    results = {}
    for mode in get_args(SearchMode):
        rows = []
        for q in questions:
            chunks = await search_chunks(pool, embedder, q["question"], CHUNKS_PER_QUERY, mode)
            ranked = top_documents([c.doc_id for c in chunks])
            found, rr = recall_and_rr(ranked, set(q["expected"]))
            rows.append({**q, "ranked": ranked, "found": found, "rr": rr})
        results[mode] = rows
    return results


def report(results: dict) -> tuple[str, bool]:
    """Markdown table (also pasted into README) and whether every mode passes MIN_RECALL."""
    modes = list(results)
    lines = [
        f"| Режим | recall@{K} | MRR@{K} | коди: recall@{K} |",
        "|---|---|---|---|",
    ]
    ok = True
    for mode in modes:
        rows = results[mode]
        recall = sum(r["found"] for r in rows) / len(rows)
        mrr = sum(r["rr"] for r in rows) / len(rows)
        codes = [r for r in rows if r["kind"] == "code"]
        code_recall = f"{sum(r['found'] for r in codes)}/{len(codes)}"
        lines.append(f"| {mode} | {recall:.2f} | {mrr:.2f} | {code_recall} |")
        ok = ok and recall >= MIN_RECALL
    lines += ["", "| Питання | " + " | ".join(f"{m}: ранг" for m in modes) + " |"]
    lines.append("|---|" + "---|" * len(modes))
    for n, q in enumerate(results[modes[0]]):
        ranks = []
        for mode in modes:
            r = results[mode][n]
            ranks.append(str(round(1 / r["rr"])) if r["rr"] else "—")
        lines.append(f"| {q['question']} | " + " | ".join(ranks) + " |")
    return "\n".join(lines), ok


async def main() -> int:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    settings = get_settings()
    name = f"eval_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(settings.database_url)
    await admin.execute(f'CREATE DATABASE "{name}"')
    try:
        url = urlunparse(urlparse(settings.database_url)._replace(path=f"/{name}"))
        pool = await create_pool(url)  # applies migrations
        try:
            embedder = OnnxEmbedder(settings.embedding_model)
            docs = load_folder(CORPUS)
            for _, doc in docs:
                await ingest_document(pool, embedder, doc)
            print(f"ingested {len(docs)} documents into {name}; {len(questions)} questions\n")
            table, ok = report(await evaluate(pool, embedder, questions))
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()
    print(table)
    if not ok:
        print(f"\nFAIL: recall@{K} below {MIN_RECALL} in at least one mode", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
