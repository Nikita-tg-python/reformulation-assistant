-- Initial schema. Idempotent: safe to run on every application start.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id      TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  doc_type    TEXT NOT NULL CHECK (doc_type IN ('ingredient_spec','trial_report','guideline')),
  content     TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id          BIGSERIAL PRIMARY KEY,
  doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INT  NOT NULL,
  text        TEXT NOT NULL,
  embedding   VECTOR(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS reformulation_runs (
  id          BIGSERIAL PRIMARY KEY,
  request     JSONB NOT NULL,
  response    JSONB,
  trace       JSONB NOT NULL DEFAULT '[]',
  status      TEXT NOT NULL,
  duration_ms INT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
