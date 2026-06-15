"""Exception hierarchy + FastAPI exception handlers + X-Request-ID middleware."""

import contextvars
import logging
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("orpheus_api")

# Contextvar so log records and downstream code can read the current id.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def get_request_id() -> str:
    return _request_id.get() or "no-request-id"


# ----- Exceptions -----

class APIError(Exception):
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail or self.__class__.__name__


class InvalidSpeakerError(APIError):
    status_code = 400
    error_code = "invalid_speaker"


class UnknownLanguageError(APIError):
    status_code = 400
    error_code = "unknown_language"


class InvalidSpeakerForLanguageError(APIError):
    status_code = 400
    error_code = "invalid_speaker_for_language"


class UpstreamUnavailableError(APIError):
    status_code = 502
    error_code = "upstream_unavailable"


class UpstreamTimeoutError(APIError):
    status_code = 504
    error_code = "upstream_timeout"


class StorageUnavailableError(APIError):
    status_code = 502
    error_code = "storage_unavailable"


# ----- Middleware -----

async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable]
):
    incoming = request.headers.get("x-request-id")
    rid = incoming or uuid.uuid4().hex
    request.state.request_id = rid
    token = _request_id.set(rid)
    try:
        response = await call_next(request)
    finally:
        _request_id.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# ----- Handlers -----

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        rid = getattr(request.state, "request_id", "no-request-id")
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail.get("error", "http_error"),
                    "detail": exc.detail.get("detail", ""),
                    "request_id": rid,
                },
                headers={"X-Request-ID": rid},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "detail": str(exc.detail),
                "request_id": rid,
            },
            headers={"X-Request-ID": rid},
        )

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError):
        rid = getattr(request.state, "request_id", "no-request-id")
        logger.warning(
            "api_error",
            extra={"request_id": rid, "code": exc.error_code, "detail": exc.detail},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "detail": exc.detail, "request_id": rid},
            headers={"X-Request-ID": rid},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        rid = getattr(request.state, "request_id", "no-request-id")
        detail = _format_validation_errors(exc)
        logger.warning(
            "validation_error",
            extra={"request_id": rid, "detail": detail},
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_request",
                "detail": detail,
                "request_id": rid,
            },
            headers={"X-Request-ID": rid},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", "no-request-id")
        logger.exception("unhandled_exception", extra={"request_id": rid})
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
                "request_id": rid,
            },
            headers={"X-Request-ID": rid},
        )


def _format_validation_errors(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "validation failed"
