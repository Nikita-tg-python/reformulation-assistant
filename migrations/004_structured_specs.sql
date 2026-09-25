-- Structured facts of ingredient specs, parsed at ingest (app/specs.py). Idempotent.
-- nutrients: [{"variant": name or null, "per_100g": {"kcal": .., "protein_g": .., ...}}]
-- allergens: EU codes (Regulation 1169/2011, Annex II) of all variants together.
-- NULL for other document types and for specs ingested before this migration.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS nutrients JSONB;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS allergens TEXT[];
