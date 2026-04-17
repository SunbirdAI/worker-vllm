# Local dev helpers for the integrated frontend + backend app.
#
# Usage:
#   make install   # one-time: npm install + uv sync
#   make web       # build the Next.js static frontend (produces web/out/)
#   make serve     # build frontend, then run uvicorn with --reload on :8000
#   make dev-web   # hot-reload UI on :3000 (proxies API to :8000 via rewrites)
#   make dev-api   # backend only on :8000 with --reload (no static mount)
#   make clean     # remove generated frontend artifacts

PORT ?= 8000
HOST ?= 0.0.0.0

.DEFAULT_GOAL := help

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## One-time: npm install (in web/) + uv sync
	cd web && npm install
	uv sync

.PHONY: web
web: ## Build the Next.js static frontend → web/out/
	cd web && npm run build

.PHONY: serve
serve: web ## Build frontend, then run uvicorn with --reload on $(PORT)
	uv run uvicorn fastapi_app:app --host $(HOST) --port $(PORT) --reload

.PHONY: dev-api
dev-api: ## Backend only with --reload (no static mount)
	uv run uvicorn fastapi_app:app --host $(HOST) --port $(PORT) --reload

.PHONY: dev-web
dev-web: ## Next.js dev server on :3000 (proxies API to http://localhost:$(PORT))
	cd web && BACKEND_URL=http://localhost:$(PORT) npm run dev

.PHONY: clean
clean: ## Remove generated frontend artifacts
	rm -rf web/.next web/out
