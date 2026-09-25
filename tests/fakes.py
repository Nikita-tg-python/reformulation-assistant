"""Test doubles shared by the test modules."""

import hashlib
import re
from typing import Any

import numpy as np

from app.chunking import Span
from app.embeddings import EMBEDDING_DIM


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words get similar vectors.

    Enough for retrieval tests without the ONNX model: a query about "аквафаба" lands on
    the document that mentions it. Tokens are whitespace-separated words.
    """

    def token_spans(self, text: str) -> list[Span]:
        return [m.span() for m in re.finditer(r"\S+", text)]

    async def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec + 1 / np.sqrt(EMBEDDING_DIM)


class _FakeConn:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
        return await self._pool.fetchval(query, *args)


class _Acquire:
    def __init__(self, pool: "FakePool") -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakeConn:
        if not self._pool.healthy:
            raise OSError("database is down")
        return _FakeConn(self._pool)

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakePool:
    """Answers only the queries /health, /ask and search_knowledge_base make."""

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        healthy: bool = True,
        documents: dict[str, str] | None = None,
        specs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.chunks = chunks or []  # dicts with doc_id, title, text, score
        self.healthy = healthy
        self.documents = documents or {}  # ingredient_spec doc_id -> content
        self.specs = specs or []  # rows of retrieval.spec_catalog: doc_id, title, nutrients

    def acquire(self, **kwargs: Any) -> _Acquire:
        return _Acquire(self)

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "EXISTS" in query:  # has_chunks; check first, it also contains "SELECT 1"
            return bool(self.chunks)
        if "SELECT 1" in query:
            return 1
        raise AssertionError(f"FakePool: unexpected query {query!r}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if "nutrients IS NOT NULL" in query:  # retrieval.spec_catalog
            return self.specs
        if "FROM documents" in query:  # retrieval.spec_facts
            return [
                {"doc_id": d, "content": c, "allergens": None}
                for d, c in self.documents.items()
                if d in args[0]
            ]
        if "websearch_to_tsquery" in query:  # full-text half of hybrid search: no matches
            return []
        top_k = args[1]
        ranked = sorted(self.chunks, key=lambda c: -c["score"])[:top_k]
        return [{"id": n, **c} for n, c in enumerate(ranked)]
