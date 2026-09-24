import json
import logging
import re
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ValidationError

from app.deps import get_embedder, get_llm, get_pool
from app.embeddings import Embedder
from app.errors import AppError
from app.llm.base import LLMClient, LLMError, Message
from app.retrieval import RetrievedChunk, has_chunks, search_chunks
from app.schemas import AskRequest, AskResponse, AskSource, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])

NO_DATA_ANSWER = "В базі знань немає даних для відповіді на це питання."

SYSTEM_PROMPT = """\
You answer questions from a food R&D team using ONLY the knowledge-base excerpts
in the user message.

Rules:
- Use only facts stated in the excerpts. No outside knowledge, no guesses, no filling gaps.
- If the excerpts do not answer the question, set "found" to false and leave "answer" empty.
- Mention the doc_id of every document you rely on inline, e.g. (SPEC-011),
  and list them in "doc_ids".
- Answer in the language of the question, concisely, keeping concrete numbers from the excerpts.

Reply with a JSON object only: {"found": boolean, "answer": string, "doc_ids": [string]}"""


class _Draft(BaseModel):
    found: bool
    answer: str = ""
    doc_ids: list[str] = []


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={
        409: {"model": ErrorResponse, "description": "empty_knowledge_base"},
        502: {"model": ErrorResponse, "description": "LLM provider error"},
        503: {"model": ErrorResponse, "description": "LLM not configured or rate limited"},
    },
    summary="Answer a question from the knowledge base, with sources",
)
async def ask(
    req: AskRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> AskResponse:
    if not await has_chunks(pool):
        raise AppError(
            409,
            "empty_knowledge_base",
            "Knowledge base is empty: ingest documents first "
            "(POST /documents or python -m app.ingest data/corpus/)",
        )
    chunks = await search_chunks(pool, embedder, req.question, req.top_k)

    raw = await llm.complete(
        [Message("system", SYSTEM_PROMPT), Message("user", _user_prompt(req.question, chunks))],
        json_output=True,
    )
    draft = _parse_draft(raw)

    # The refusal text is fixed in code, not left to the model's wording.
    if not draft.found or not draft.answer.strip():
        return AskResponse(answer=NO_DATA_ANSWER, sources=[])

    cited = set(draft.doc_ids) & {c.doc_id for c in chunks}
    if not cited:
        logger.warning(
            "answer cites no retrieved doc_id (got %s); returning all chunks", draft.doc_ids
        )
    used = [c for c in chunks if not cited or c.doc_id in cited]
    return AskResponse(
        answer=draft.answer.strip(),
        sources=[
            AskSource(doc_id=c.doc_id, title=c.title, chunk_text=c.text, score=c.score)
            for c in used
        ],
    )


def _user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n---\n\n".join(f"[{c.doc_id}] {c.title}\n{c.text}" for c in chunks)
    return f"Knowledge-base excerpts:\n\n{excerpts}\n\n---\n\nQuestion: {question}"


def _parse_draft(raw: str) -> _Draft:
    # Some models wrap JSON in ```json fences even in JSON mode.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return _Draft.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("unparseable LLM answer: %r", raw[:500])
        raise LLMError("LLM returned an answer that is not valid JSON", "llm_bad_response") from exc
