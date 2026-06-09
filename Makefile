# finrag developer commands. Backend tasks run through uv; stack tasks through
# Docker Compose. Run `make help` for the list.

COMPOSE := docker-compose -f infra/compose.yaml
BACKEND := backend
FRONTEND := frontend
OLLAMA_MODEL := qwen2.5:7b-instruct
API_URL := http://localhost:8000
# Port the Next dev server binds (see frontend/package.json "dev"); frontend-down
# stops whatever holds it.
FE_PORT := 3001

# Data-store handles for clean-data (defaults match infra/compose.yaml).
COMPOSE_NET := finrag_finrag-net
QDRANT_URL := http://localhost:6333
INGEST_TOPIC := finrag-ingest
INGEST_BUCKET := filings
PG_USER := finrag
PG_DB := finrag
MINIO_USER := finrag
MINIO_PASS := finrag-secret

.DEFAULT_GOAL := help
.PHONY: help \
	up down dev backend-up backend-down frontend-up frontend-down \
	logs ps consumer-logs model clean clean-data \
	install fmt lint typecheck test check

# Help groups targets by `##@ Section` headers and prints each target's `## doc`.
# Targets below are ordered by how they're used (each start/stop pair adjacent),
# since this listing follows file order.
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@ /{printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} \
		/^[a-zA-Z_-]+:.*## /{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ App lifecycle — start/stop the whole app, or each half on its own
# The backend is the Docker Compose stack; the frontend is a host-run Next dev
# server (no container). `up`/`down` drive both; `backend-*`/`frontend-*` each half.
up: backend-up frontend-up ## Start the whole app — backend stack + frontend
down: frontend-down backend-down ## Stop the whole app — frontend + backend stack
dev: up ## Alias for `up`

backend-up: ## Start the backend stack (detached); rebuilds the app image
	$(COMPOSE) up -d --build

backend-down: ## Stop the backend stack
	$(COMPOSE) down

# The frontend runs in the foreground so its HMR logs are visible (Ctrl-C stops
# it); frontend-down is the out-of-band stop (orphan, or `down` from another shell).
frontend-up: ## Start the frontend dev server (foreground; Ctrl-C to stop)
	cd $(FRONTEND) && FINRAG_API_URL=$(API_URL) pnpm dev

frontend-down: ## Stop the frontend dev server (kills the holder of FE_PORT)
	@pids=$$(lsof -ti tcp:$(FE_PORT) 2>/dev/null | tr '\n' ' '); \
	if [ -n "$$pids" ]; then echo "→ stopping frontend on :$(FE_PORT) (pid $$pids)"; kill $$pids; \
	else echo "frontend not running on :$(FE_PORT)"; fi

##@ Stack ops & data
logs: ## Tail backend stack logs
	$(COMPOSE) logs -f

ps: ## Show backend stack status
	$(COMPOSE) ps

consumer-logs: ## Tail the ingestion consumer (the finrag-consumer service auto-drains)
	$(COMPOSE) logs -f finrag-consumer

model: ## Pull the LLM model into Ollama (run once after `up`)
	$(COMPOSE) exec finrag-ollama ollama pull $(OLLAMA_MODEL)

clean: ## Stop the stack and remove volumes
	$(COMPOSE) down -v

clean-data: ## Wipe ingested data (Postgres/Qdrant/MinIO/topic); keeps the model + HF cache
	@echo "→ Postgres: truncating facts + conversation tables"
	$(COMPOSE) exec -T finrag-postgres psql -U $(PG_USER) -d $(PG_DB) -c \
		"TRUNCATE financial_facts, filings, collections, concept_index, ingested_documents, quarantine, ingestion_documents, ingestion_trace, chat_sessions, chat_messages CASCADE"
	@echo "→ Qdrant: deleting vector collections (recreated on next ingest)"
	-curl -fsS -X DELETE $(QDRANT_URL)/collections/finrag_chunks >/dev/null
	-curl -fsS -X DELETE $(QDRANT_URL)/collections/finrag_eval >/dev/null
	@echo "→ MinIO: emptying the $(INGEST_BUCKET) bucket"
	docker run --rm --network $(COMPOSE_NET) --entrypoint sh minio/mc:latest -c \
		"mc alias set m http://finrag-minio:9000 $(MINIO_USER) $(MINIO_PASS) >/dev/null && \
		 mc rm --recursive --force m/$(INGEST_BUCKET) >/dev/null 2>&1 || true"
	@echo "→ Redpanda: purging the $(INGEST_TOPIC) topic"
	$(COMPOSE) exec -T finrag-redpanda sh -c \
		"rpk topic delete $(INGEST_TOPIC) 2>/dev/null; sleep 2; rpk topic create $(INGEST_TOPIC) 2>/dev/null; true"
	@echo "→ Restarting API + consumer (fresh clients, recreate collections)"
	$(COMPOSE) restart finrag-api finrag-consumer
	@echo "✓ Data cleared; Ollama model + HF cache preserved."

##@ Backend code — lint / type / test (run through uv)
install: ## Install backend deps into a local venv
	cd $(BACKEND) && uv sync

fmt: ## Format backend code
	cd $(BACKEND) && uv run ruff format .

lint: ## Lint backend code
	cd $(BACKEND) && uv run ruff check .

typecheck: ## Type-check backend code
	cd $(BACKEND) && uv run mypy src

test: ## Run backend tests
	cd $(BACKEND) && uv run pytest

check: lint typecheck test ## Run lint, types, and tests
