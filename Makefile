# All targets run in Docker: the host needs only Docker and make.
.PHONY: help up down logs ingest test lint fmt clean k8s-secrets k8s-up ingest-k8s k8s-down

TEST = docker compose --profile dev run --rm test sh -c

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

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

# ---------- Kubernetes on kind (k8s/, see README) ----------

KIND_CLUSTER = reformulation
K8S_IMAGE = reformulation-assistant:local
KUBECTL = kubectl -n reformulation

k8s-secrets:  ## create k8s/secrets.env: placeholders + API keys from .env (nothing printed)
	@cp k8s/secrets.env.example k8s/secrets.env
	@if [ -f .env ]; then grep -E '^(GROQ|GEMINI)_API_KEY=.+' .env >> k8s/secrets.env || true; fi
	@echo "k8s/secrets.env: keys set for $$(grep -oE '^(GROQ|GEMINI)_API_KEY' k8s/secrets.env | tr '\n' ' ')"

k8s-up: k8s-secrets  ## kind cluster + image + kubectl apply -k k8s/ (ingest Job included), wait
	@kind get clusters | grep -qx $(KIND_CLUSTER) || kind create cluster --name $(KIND_CLUSTER)
	docker build -t $(K8S_IMAGE) .
	kind load docker-image $(K8S_IMAGE) --name $(KIND_CLUSTER)
	-@$(KUBECTL) delete job ingest --ignore-not-found 2>/dev/null
	kubectl apply -k k8s/
	$(KUBECTL) rollout status statefulset/postgres --timeout=180s
	$(KUBECTL) rollout status deployment/api --timeout=300s

ingest-k8s:  ## rerun the ingest Job in the kind cluster (idempotent), print its summary
	$(KUBECTL) delete job ingest --ignore-not-found
	kubectl apply -k k8s/ | grep ingest
	$(KUBECTL) wait --for=condition=complete job/ingest --timeout=300s
	$(KUBECTL) logs job/ingest | tail -1

k8s-down:  ## delete the kind cluster with all its data
	kind delete cluster --name $(KIND_CLUSTER)
