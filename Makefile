.PHONY: help test test-all lint install docker-up docker-down run clean

PROJECTS := 01-nl-stock-query 02-rag-financial-kb 03-ai-strategy-generator 04-strategy-diagnostics 05-finllm-finetune

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Testing ─────────────────────────────────────────────
test: ## Run tests for a single project (usage: make test P=01-nl-stock-query)
	@if [ -z "$(P)" ]; then echo "Usage: make test P=<project-name>"; exit 1; fi
	cd projects/$(P) && PYTHONPATH=src .venv/bin/pytest tests/ -v --tb=short

test-all: ## Run all 594 tests across all projects
	@total=0; pass=0; fail=0; \
	for p in $(PROJECTS); do \
		echo "\n\033[1m=== $$p ===\033[0m"; \
		cd projects/$$p && \
		result=$$(PYTHONPATH=src .venv/bin/pytest tests/ -q --tb=line 2>&1) && \
		echo "$$result" | tail -1; \
		cd ../..; \
	done

test-quick: ## Run tests in quiet mode (summary only)
	@for p in $(PROJECTS); do \
		cd projects/$$p && \
		printf "%-35s " "$$p:" && \
		PYTHONPATH=src .venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -1; \
		cd ../..; \
	done

# ── Code Quality ────────────────────────────────────────
lint: ## Run ruff linter on all source code
	ruff check projects/*/src/ --select E,F,W --ignore E501,E741,F841 --fix

format: ## Auto-format all source code
	ruff format projects/*/src/

# ── Install ─────────────────────────────────────────────
install: ## Install all project venvs + dependencies
	@for p in $(PROJECTS); do \
		echo "\n\033[1mInstalling $$p...\033[0m"; \
		cd projects/$$p && \
		python3 -m venv .venv && \
		.venv/bin/pip install --upgrade pip -q && \
		.venv/bin/pip install -r requirements.txt -q && \
		cd ../..; \
	done
	@echo "\n\033[32mAll projects installed.\033[0m"

# ── Docker ──────────────────────────────────────────────
docker-build: ## Build Docker images for all API services
	docker-compose build

docker-up: ## Start all API services
	docker-compose up -d
	@echo "\n\033[32mServices starting:\033[0m"
	@echo "  P1 API:  http://localhost:8001/docs"
	@echo "  P2 API:  http://localhost:8002/docs"
	@echo "  P3 API:  http://localhost:8003/docs"
	@echo "  P4 API:  http://localhost:8004/docs"
	@echo "  P5 API:  http://localhost:8005/docs"

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## Tail logs from all services
	docker-compose logs -f

# ── Run individual services ─────────────────────────────
run: ## Run a project's API server (usage: make run P=01-nl-stock-query PORT=8001)
	@if [ -z "$(P)" ]; then echo "Usage: make run P=<project> [PORT=8001]"; exit 1; fi
	cd projects/$(P) && \
	PYTHONPATH=src .venv/bin/uvicorn src.api:app --host 0.0.0.0 --port $${PORT:-8001} --reload

# ── Cleanup ─────────────────────────────────────────────
clean: ## Remove all __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "\033[32mCleaned.\033[0m"

# ── Stats ───────────────────────────────────────────────
stats: ## Show project statistics
	@echo "\033[1m📊 Project Statistics\033[0m"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@printf "%-35s %8s %8s\n" "Project" "Files" "Lines"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@for p in $(PROJECTS); do \
		files=$$(find projects/$$p/src -name "*.py" | wc -l); \
		lines=$$(find projects/$$p/src -name "*.py" -exec cat {} + | wc -l); \
		printf "%-35s %8d %8d\n" "$$p" "$$files" "$$lines"; \
	done
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@printf "%-35s %8d %8d\n" "TOTAL" \
		$$(find projects/*/src -name "*.py" | wc -l) \
		$$(find projects/*/src -name "*.py" -exec cat {} + | wc -l)
