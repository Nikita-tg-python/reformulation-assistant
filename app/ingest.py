"""Document ingestion: shared logic for POST /documents and the CLI.

CLI: python -m app.ingest data/corpus/
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import asyncpg
from pydantic import ValidationError

from app.chunking import chunk_text
from app.config import get_settings
from app.db import create_pool
from app.embeddings import Embedder, OnnxEmbedder
from app.schemas import DocumentIn
from app.specs import parse_allergens, parse_nutrients

_FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n(.*)\Z", re.DOTALL)


async def ingest_document(pool: asyncpg.Pool, embedder: Embedder, doc: DocumentIn) -> int:
    """Store the document and replace all of its chunks. Returns the number of chunks.

    Idempotent per doc_id: re-ingesting replaces old chunks instead of adding duplicates.
    """
    chunks = chunk_text(doc.content, embedder.token_spans)
    # Embed before opening the transaction so row locks aren't held during CPU work.
    vectors = await embedder.embed_passages(chunks) if chunks else []
    is_spec = doc.doc_type == "ingredient_spec"
    nutrients = parse_nutrients(doc.content) if is_spec else None
    allergens = parse_allergens(doc.content) if is_spec else None

    async with pool.acquire() as conn, conn.transaction():
        # Upsert first: its row lock serialises concurrent ingests of the same doc_id,
        # so the DELETE below always sees the other transaction's committed chunks.
        await conn.execute(
            """
            INSERT INTO documents (doc_id, title, doc_type, content, nutrients, allergens)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            ON CONFLICT (doc_id) DO UPDATE
               SET title = EXCLUDED.title,
                   doc_type = EXCLUDED.doc_type,
                   content = EXCLUDED.content,
                   nutrients = EXCLUDED.nutrients,
                   allergens = EXCLUDED.allergens
            """,
            doc.doc_id,
            doc.title,
            doc.doc_type,
            doc.content,
            json.dumps(nutrients, ensure_ascii=False) if nutrients is not None else None,
            allergens,
        )
        await conn.execute("DELETE FROM chunks WHERE doc_id = $1", doc.doc_id)
        await conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, embedding) VALUES ($1, $2, $3, $4)",
            [
                (doc.doc_id, i, text, vec)
                for i, (text, vec) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
    return len(chunks)


def parse_markdown(text: str) -> DocumentIn:
    """Parse a corpus file: flat `key: value` frontmatter (doc_id, title, doc_type) + body."""
    match = _FRONTMATTER.match(text.replace("\r\n", "\n"))
    if not match:
        raise ValueError("missing frontmatter block delimited by '---'")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"bad frontmatter line: {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        meta[key.strip()] = value
    return DocumentIn(**meta, content=match.group(2))


def load_folder(folder: Path) -> list[tuple[Path, DocumentIn]]:
    """Parse every *.md file. Raises ValueError listing all bad files, so nothing is
    ingested unless the whole folder is valid (no half-ingested corpus)."""
    files = sorted(folder.glob("*.md"))
    if not files:
        raise ValueError(f"no .md files in {folder}")
    docs, errors = [], []
    for path in files:
        try:
            docs.append((path, parse_markdown(path.read_text(encoding="utf-8"))))
        except (ValueError, ValidationError) as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        raise ValueError("invalid files, nothing ingested:\n  " + "\n  ".join(errors))
    return docs


async def ingest_all(docs: list[tuple[Path, DocumentIn]]) -> None:
    settings = get_settings()
    embedder = OnnxEmbedder(settings.embedding_model)
    pool = await create_pool(settings.database_url)
    try:
        total = 0
        for path, doc in docs:
            n = await ingest_document(pool, embedder, doc)
            total += n
            print(f"{doc.doc_id:<10} {n:>3} chunks  {path.name}")
        print(f"ingested {len(docs)} documents, {total} chunks")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest", description="Ingest a folder of markdown documents."
    )
    parser.add_argument("folder", type=Path, help="folder with *.md files, e.g. data/corpus/")
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error(f"not a directory: {args.folder}")
    try:
        docs = load_folder(args.folder)
    except ValueError as exc:
        sys.exit(str(exc))
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(ingest_all(docs))


if __name__ == "__main__":
    main()
