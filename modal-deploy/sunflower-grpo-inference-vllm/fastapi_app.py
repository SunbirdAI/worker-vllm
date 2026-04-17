"""
Standalone FastAPI proxy in front of the deployed Modal `web` endpoint.

Why a proxy?
    The Modal app already exposes /generate, /generate_stream, /health at
    https://<workspace>--sunflower-grpo-vllm-web.modal.run. This wrapper sits
    in front of it so we can add auth, rate-limiting, logging, custom domains,
    request shaping, etc. without touching the inference deployment.

Run locally:
    uv add fastapi uvicorn httpx
    uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

Configuration (env vars):
    SUNFLOWER_UPSTREAM_URL  Upstream Modal web URL (no trailing slash).
                            Default: https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run
    UPSTREAM_TIMEOUT        Per-request timeout in seconds. Default: 600.
    SUNBIRD_PROD_URL        Production Sunbird AI API base URL.
                            Default: https://api.sunbird.ai
    SUNBIRD_PROD_TOKEN      Bearer token for the production API (required for
                            /generate_production).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPSTREAM_URL = os.environ.get(
    "SUNFLOWER_UPSTREAM_URL",
    "https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run",
).rstrip("/")
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "600"))
SUNBIRD_PROD_URL = os.environ.get("SUNBIRD_PROD_URL", "https://api.sunbird.ai").rstrip("/")
SUNBIRD_PROD_TOKEN = os.environ.get("SUNBIRD_PROD_TOKEN")
RATE_LIMIT_PER_MINUTE = os.environ.get("RATE_LIMIT_PER_MINUTE", "100/minute")
RATE_LIMIT_PER_DAY = os.environ.get("RATE_LIMIT_PER_DAY", "1000/day")
# Optional Redis-backed storage for multi-process / multi-replica deploys.
RATE_LIMIT_STORAGE_URI = os.environ.get("RATE_LIMIT_STORAGE_URI", "memory://")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=RATE_LIMIT_STORAGE_URI,
    default_limits=[RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_DAY],
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("sunflower-proxy")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    instruction: str = Field(..., description="User prompt / instruction")
    temperature: float = Field(0.6, ge=0.0, le=1.0)
    max_tokens: int = Field(1024, ge=1, le=4096)


class GenerateResponse(BaseModel):
    response: str


class ProductionRequest(BaseModel):
    instruction: str = Field(..., description="User prompt / instruction")
    model_type: str = Field("qwen", description="Production model type, e.g. `qwen`")
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    system_message: str = Field("", description="Optional system message override")


# ---------------------------------------------------------------------------
# Lifespan: shared httpx client
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        base_url=UPSTREAM_URL,
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=15.0),
    )
    app.state.prod_http = httpx.AsyncClient(
        base_url=SUNBIRD_PROD_URL,
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT, connect=15.0),
    )
    log.info(
        "Proxy ready, upstream=%s prod_upstream=%s",
        UPSTREAM_URL,
        SUNBIRD_PROD_URL,
    )
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.prod_http.aclose()


app = FastAPI(
    title="Sunflower GRPO Inference Proxy",
    description="HTTP proxy in front of the Modal-hosted Sunflower vLLM deployment.",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting (slowapi) -----------------------------------------------------
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "limits": [RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_DAY],
        },
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)


# ---------------------------------------------------------------------------
# Middleware: request id + access log
# ---------------------------------------------------------------------------
@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    log.info("→ %s %s rid=%s", request.method, request.url.path, rid)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("✗ unhandled error rid=%s", rid)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = rid
    log.info(
        "← %s %s status=%d rid=%s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        rid,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness for the proxy + readiness probe of the upstream."""
    client: httpx.AsyncClient = app.state.http
    upstream_status: str
    try:
        r = await client.get("/health", timeout=10.0)
        upstream_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        upstream_status = f"unreachable: {e.__class__.__name__}"
    return {
        "status": "ok",
        "upstream": UPSTREAM_URL,
        "upstream_status": upstream_status,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    client: httpx.AsyncClient = app.state.http
    try:
        r = await client.post("/generate", json=req.model_dump())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest):
    client: httpx.AsyncClient = app.state.http

    async def passthrough() -> AsyncIterator[bytes]:
        try:
            async with client.stream(
                "POST", "/generate_stream", json=req.model_dump()
            ) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    raise HTTPException(status_code=upstream.status_code, detail=body.decode())
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
        except httpx.HTTPError as e:
            log.exception("upstream stream error")
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

    return StreamingResponse(passthrough(), media_type="text/event-stream")


@app.post("/generate_production")
async def generate_production(req: ProductionRequest):
    """Forward to the production Sunbird AI API for side-by-side comparison.

    The upstream expects form-encoded (`application/x-www-form-urlencoded`)
    data with a bearer token distinct from this proxy's own API_KEY.
    """
    if not SUNBIRD_PROD_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="SUNBIRD_PROD_TOKEN is not configured on the proxy",
        )
    client: httpx.AsyncClient = app.state.prod_http
    try:
        r = await client.post(
            "/tasks/sunflower_simple",
            headers={
                "accept": "application/json",
                "Authorization": f"Bearer {SUNBIRD_PROD_TOKEN}",
            },
            data={
                "instruction": req.instruction,
                "model_type": req.model_type,
                "temperature": str(req.temperature),
                "system_message": req.system_message,
            },
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502, detail=f"Production upstream error: {e}"
        ) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


# ---------------------------------------------------------------------------
# Static frontend (Next.js static export → web/out/). Mounted last so all
# explicit API routes above take precedence over the SPA fallback.
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "web" / "out"
if _STATIC_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=_STATIC_DIR, html=True),
        name="frontend",
    )
    log.info("Mounted static frontend from %s", _STATIC_DIR)
else:
    log.info(
        "No static frontend at %s — run `npm run build` in web/ to generate it.",
        _STATIC_DIR,
    )
