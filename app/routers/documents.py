from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends

from app.deps import get_embedder, get_pool
from app.embeddings import Embedder
from app.ingest import ingest_document
from app.schemas import DocumentIn, DocumentIngestResponse, ErrorResponse

router = APIRouter(tags=["documents"])


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    responses={422: {"model": ErrorResponse}},
    summary="Ingest one document (re-ingesting a doc_id replaces its chunks)",
)
async def create_document(
    doc: DocumentIn,
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> DocumentIngestResponse:
    chunks_created = await ingest_document(pool, embedder, doc)
    return DocumentIngestResponse(doc_id=doc.doc_id, chunks_created=chunks_created)
