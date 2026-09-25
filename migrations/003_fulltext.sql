-- Full-text search for hybrid retrieval (vector + tsvector, fused with RRF). Idempotent.
-- 'simple' config: no stemming and no stop words, so codes like SPEC-001, E-numbers and
-- exact terms match as written (Postgres has no Ukrainian dictionary).
-- doc_id is included with weight A: a chunk never contains its own document code
-- (only other documents cite it), so without it "SPEC-001" would find the citing documents.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', doc_id), 'A') || to_tsvector('simple', text)
  ) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
