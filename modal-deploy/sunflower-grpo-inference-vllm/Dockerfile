# syntax=docker/dockerfile:1.6

# ---------- Stage 1: build the Next.js static frontend ----------
FROM node:22-slim AS web-builder

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /web

COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY web/ ./
# Produces /web/out via `BUILD_MODE=export next build`.
RUN npm run build


# ---------- Stage 2: python deps via uv ----------
FROM python:3.12-slim AS py-builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY fastapi_app.py ./


# ---------- Stage 3: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8080

RUN groupadd --system --gid 1001 app \
 && useradd  --system --uid 1001 --gid app --home-dir /app --shell /sbin/nologin app

WORKDIR /app

COPY --from=py-builder  --chown=app:app /app/.venv          /app/.venv
COPY --from=py-builder  --chown=app:app /app/fastapi_app.py /app/fastapi_app.py
COPY --from=web-builder --chown=app:app /web/out            /app/web/out

USER app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT}"]
