# Stage 1: dependencies and the embedding model. Stage 2: only venv, model and code.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# Bake the model into the image so containers never download it at start.
# Only the files app/embeddings.py reads: ONNX weights, tokenizer, sentence-transformers config.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${EMBEDDING_MODEL}', allow_patterns=['onnx/model.onnx', 'tokenizer.json', 'sentence_bert_config.json'])"


FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/hf /opt/hf
COPY app ./app
COPY migrations ./migrations
COPY data ./data

USER app
EXPOSE 8000
# Access log off: the app middleware logs every request as JSON with its request_id.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
