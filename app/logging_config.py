"""Structured JSON logs: one formatter for every logger, request_id on every line.

The request id lives in a ContextVar set by the HTTP middleware in app.main. asyncio
copies context into tasks, so logs from the agent loop, tools and LLM clients carry the
id of the request that triggered them without passing it around explicitly.
"""

import json
import logging
import re
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_INCOMING_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # accept only log-safe client ids
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def new_request_id(incoming: str | None) -> str:
    return incoming if incoming and _INCOMING_ID.match(incoming) else uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # Fields passed via logger.info("...", extra={...}) become top-level keys.
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn installs its own text handlers before importing the app; route them to ours.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # The request middleware in app.main logs every request with its request_id instead.
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
