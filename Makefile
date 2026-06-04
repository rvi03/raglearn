# raglearn developer commands. Backend tasks run through uv; stack tasks through
# Docker Compose. Run `make help` for the list.

COMPOSE := docker compose -f infra/compose.yaml
BACKEND := backend

.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check up down logs ps clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# Backend
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

# Stack
up: ## Build and start the full stack
	$(COMPOSE) up -d --build

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail stack logs
	$(COMPOSE) logs -f

ps: ## Show stack status
	$(COMPOSE) ps

clean: ## Stop the stack and remove volumes
	$(COMPOSE) down -v
