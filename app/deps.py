"""FastAPI dependencies for process-wide resources created in the lifespan."""

import asyncpg
import httpx
from fastapi import Request

from app.embeddings import Embedder
from app.llm.base import LLMClient


def get_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pool


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http
