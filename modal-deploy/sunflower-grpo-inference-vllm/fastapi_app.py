"""
Standalone FastAPI app that serves Sunflower GRPO inference by proxying to the
deployed Modal class (`sunflower-grpo-vllm` / `SunflowerVLLM`).

Run locally:
    uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health            -> {"status": "ok"}
    POST /generate          -> {"response": "..."}
    POST /generate_stream   -> text/event-stream (SSE) of {"delta": "..."} chunks

Request body (JSON) for /generate and /generate_stream:
    {
        "instruction": "Translate to luganda: Good morning",
        "temperature": 0.6,        # optional, default 0.6
        "max_tokens":  1024        # optional, default 1024
    }
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODAL_APP_NAME = os.environ.get("MODAL_APP_NAME", "sunflower-grpo-vllm")
MODAL_CLASS_NAME = os.environ.get("MODAL_CLASS_NAME", "SunflowerVLLM")


class GenerateRequest(BaseModel):
    instruction: str = Field(..., description="User prompt / instruction")
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=4096)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve the deployed Modal class once at startup."""
    try:
        cls = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
        app.state.sunflower = cls()
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve Modal class {MODAL_APP_NAME}/{MODAL_CLASS_NAME}. "
            f"Is the app deployed and are you authenticated with `modal setup`? "
            f"Underlying error: {e}"
        ) from e
    yield


app = FastAPI(
    title="Sunflower GRPO Inference API",
    description="FastAPI proxy to the Modal-hosted vLLM deployment.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "modal_app": MODAL_APP_NAME, "modal_class": MODAL_CLASS_NAME}


@app.post("/generate")
async def generate(req: GenerateRequest):
    sunflower = app.state.sunflower
    try:
        text = await sunflower.generate.remote.aio(
            req.instruction, req.temperature, req.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream Modal error: {e}") from e
    return {"response": text}


@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest):
    sunflower = app.state.sunflower

    async def event_source():
        try:
            async for delta in sunflower.generate_stream.remote_gen.aio(
                req.instruction, req.temperature, req.max_tokens
            ):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
