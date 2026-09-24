-- Link each agent run to the request_id in the JSON logs. Idempotent.

ALTER TABLE reformulation_runs ADD COLUMN IF NOT EXISTS request_id TEXT;
CREATE INDEX IF NOT EXISTS reformulation_runs_created_at_idx ON reformulation_runs (created_at DESC);
