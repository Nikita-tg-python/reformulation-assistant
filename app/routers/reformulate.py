import json
import logging
import time
from typing import Annotated, Any

import asyncpg
import httpx
from fastapi import APIRouter, Depends

from app.agent.loop import AgentError, run_agent
from app.agent.pipeline import run_pipeline
from app.agent.tools import AgentTools
from app.config import get_settings
from app.deps import get_embedder, get_http, get_llm, get_pool
from app.embeddings import Embedder
from app.llm.base import LLMClient
from app.logging_config import request_id_var
from app.schemas import AgentErrorResponse, ReformulateRequest, ReformulateResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reformulate"])


@router.post(
    "/reformulate",
    response_model=ReformulateResponse,
    responses={
        502: {"model": AgentErrorResponse, "description": "LLM error or invalid agent output"},
        503: {"model": AgentErrorResponse, "description": "LLM not configured or rate limited"},
        504: {"model": AgentErrorResponse, "description": "agent_timeout: iteration or time limit"},
    },
    summary="Propose ingredient substitutions for a goal, with sources and trace",
)
async def reformulate(
    req: ReformulateRequest,
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    http: Annotated[httpx.AsyncClient, Depends(get_http)],
) -> ReformulateResponse:
    settings = get_settings()
    tools = AgentTools(pool, embedder, http)  # fresh per run: per-run product cache
    started = time.perf_counter()
    try:
        if settings.agent_mode == "pipeline":
            result = await run_pipeline(llm, tools, req, timeout_s=settings.agent_timeout_seconds)
        else:
            result = await run_agent(
                llm,
                tools,
                req,
                max_iterations=settings.agent_max_iterations,
                timeout_s=settings.agent_timeout_seconds,
            )
    except AgentError as exc:
        await _record_run(
            pool, req, {"error": {"code": exc.code, "message": exc.message}}, exc.trace,
            exc.code, started,
        )  # fmt: skip
        raise
    response = ReformulateResponse(**result.answer.model_dump(), trace=result.trace)
    await _record_run(pool, req, response.model_dump(mode="json"), result.trace, "ok", started)
    return response


async def _record_run(
    pool: asyncpg.Pool,
    req: ReformulateRequest,
    response: dict[str, Any],
    trace: list[dict[str, Any]],
    status: str,
    started: float,
) -> None:
    """Every run is stored, including failures. A logging failure never breaks the reply."""
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info("agent run finished", extra={"status": status, "duration_ms": duration_ms})
    try:
        await pool.execute(
            """
            INSERT INTO reformulation_runs
                   (request, response, trace, status, duration_ms, request_id)
            VALUES ($1::jsonb, $2::jsonb, $3::jsonb, $4, $5, $6)
            """,
            req.model_dump_json(),
            json.dumps(response, ensure_ascii=False),
            json.dumps(trace, ensure_ascii=False, default=str),
            status,
            duration_ms,
            request_id_var.get(),
        )
    except Exception:
        logger.exception("failed to record reformulation run")
