# All targets run in Docker: the host needs only Docker and make.
.PHONY: help up down logs ingest test lint fmt clean

TEST = docker compose --profile dev run --rm test sh -c

help:  ## list targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

up:  ## build and start api + db, wait until /health answers
	docker compose up -d --build
	@echo "waiting for http://localhost:8000/health ..."
	@for i in $$(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
	@curl -s localhost:8000/health; echo

down:  ## stop containers (data volume is kept)
	docker compose down

logs:  ## follow api logs (JSON lines)
	docker compose logs -f api

ingest:  ## load data/corpus/ into the knowledge base (idempotent)
	docker compose exec api python -m app.ingest data/corpus/

test:  ## pytest incl. integration tests against the compose db; no LLM keys, no internet
	$(TEST) "uv sync --locked -q && uv run --locked pytest -q"

lint:  ## ruff check + format check
	$(TEST) "uv sync --locked -q && uv run --locked ruff check . && uv run --locked ruff format --check ."

fmt:  ## apply ruff fixes and formatting
	$(TEST) "uv sync --locked -q && uv run --locked ruff check --fix . && uv run --locked ruff format ."

clean:  ## remove containers AND volumes (database, test venv)
	docker compose --profile dev down -v
