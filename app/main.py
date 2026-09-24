import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import check_db, create_pool
from app.embeddings import OnnxEmbedder
from app.errors import register_error_handlers
from app.llm import create_llm_client
from app.logging_config import new_request_id, request_id_var, setup_logging
from app.routers import ask, documents, reformulate
from app.schemas import ErrorResponse, HealthResponse

setup_logging()
logger = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.embedder = OnnxEmbedder(settings.embedding_model)
    app.state.llm = create_llm_client(settings)
    pool = await create_pool(settings.database_url)
    app.state.http = httpx.AsyncClient()  # shared connection pool for Open Food Facts
    try:
        app.state.pool = pool
        yield
    finally:
        await app.state.http.aclose()
        await pool.close()


app = FastAPI(
    title="Reformulation Assistant",
    version="0.1.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)
register_error_handlers(app)
app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(reformulate.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign a request id (or accept X-Request-ID), expose it, log one line per request."""
    request_id = new_request_id(request.headers.get("x-request-id"))
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        request_id_var.reset(token)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unavailable"}},
)
async def health(request: Request) -> JSONResponse:
    db_ok = await check_db(request.app.state.pool)
    body = HealthResponse(
        status="ok" if db_ok else "degraded",
        db="ok" if db_ok else "error",
        llm_provider=get_settings().llm_provider,
    )
    return JSONResponse(status_code=200 if db_ok else 503, content=body.model_dump())
