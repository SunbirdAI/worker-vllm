import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    """Mini FastAPI app wired with our handlers, used only by tests."""
    from api.errors import (
        APIError,
        InvalidSpeakerError,
        StorageUnavailableError,
        UpstreamTimeoutError,
        UpstreamUnavailableError,
        register_exception_handlers,
        request_id_middleware,
    )

    app = FastAPI()
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)

    @app.get("/raise/{kind}")
    def raise_kind(kind: str):
        if kind == "invalid_speaker":
            raise InvalidSpeakerError("speaker x not found")
        if kind == "upstream_unavailable":
            raise UpstreamUnavailableError("modal said 502")
        if kind == "upstream_timeout":
            raise UpstreamTimeoutError("modal slow")
        if kind == "storage_unavailable":
            raise StorageUnavailableError("gcs down")
        if kind == "generic":
            raise RuntimeError("boom")
        return {"ok": True}

    return app


def test_request_id_generated_if_missing():
    client = TestClient(_build_app())
    r = client.get("/raise/ok-path-doesnt-exist")  # 404 from FastAPI
    assert "x-request-id" in {k.lower() for k in r.headers}


def test_request_id_echoed_if_provided():
    client = TestClient(_build_app())
    r = client.get("/raise/ok", headers={"X-Request-ID": "client-123"})
    # 404 because /raise/ok isn't defined either, but middleware still runs
    assert r.headers.get("x-request-id") == "client-123"


@pytest.mark.parametrize(
    "kind,expected_status,expected_code",
    [
        ("invalid_speaker", 400, "invalid_speaker"),
        ("upstream_unavailable", 502, "upstream_unavailable"),
        ("upstream_timeout", 504, "upstream_timeout"),
        ("storage_unavailable", 502, "storage_unavailable"),
        ("generic", 500, "internal_error"),
    ],
)
def test_handlers_map_errors_to_responses(kind, expected_status, expected_code):
    # raise_server_exceptions=False is required so TestClient does not re-raise the
    # generic Exception case; we want to verify the handler turns it into a 500.
    client = TestClient(_build_app(), raise_server_exceptions=False)
    r = client.get(f"/raise/{kind}", headers={"X-Request-ID": "abc"})
    assert r.status_code == expected_status
    body = r.json()
    assert body["error"] == expected_code
    assert body["request_id"] == "abc"
    assert "detail" in body
    # generic error must NOT leak the underlying message
    if kind == "generic":
        assert "boom" not in body["detail"].lower()


def test_validation_error_uses_uniform_shape():
    from api.errors import register_exception_handlers, request_id_middleware
    from pydantic import BaseModel

    app = FastAPI()
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)

    class Body(BaseModel):
        x: int

    @app.post("/v")
    def v(b: Body):
        return {"ok": True}

    client = TestClient(app)
    r = client.post("/v", json={"x": "not-an-int"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_request"
    assert "request_id" in body
