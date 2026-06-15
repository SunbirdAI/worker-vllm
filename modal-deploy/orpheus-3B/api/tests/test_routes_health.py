import httpx
from fastapi.testclient import TestClient

from api.tests.conftest import SPEAKERS_PAYLOAD, build_test_app


def _healthy_handler():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        return httpx.Response(404)

    return handler


def _modal_down_handler():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        return httpx.Response(503, text="modal down")

    return handler


def test_health_liveness_no_upstream():
    app = build_test_app(_healthy_handler())
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "orpheus-tts-api"}


def test_healthz_all_upstreams_ok():
    app = build_test_app(_healthy_handler())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["upstreams"]["modal"] == "ok"
    assert body["upstreams"]["gcs"] == "ok"


def test_healthz_modal_down_returns_503():
    app = build_test_app(_modal_down_handler())
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["upstreams"]["modal"] != "ok"


def test_healthz_gcs_down_returns_503():
    app = build_test_app(_healthy_handler())
    # break the fake bucket
    app.state.storage._client.bucket_obj._exists = False
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["upstreams"]["gcs"] != "ok"
