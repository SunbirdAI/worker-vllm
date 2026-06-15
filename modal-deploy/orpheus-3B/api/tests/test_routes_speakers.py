from fastapi.testclient import TestClient

from api.tests.conftest import SPEAKERS_PAYLOAD, build_test_app


def test_get_speakers_returns_catalog(speakers_handler):
    app = build_test_app(speakers_handler)
    client = TestClient(app)
    r = client.get("/speakers")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == SPEAKERS_PAYLOAD["default"]
    assert body["by_language"] == SPEAKERS_PAYLOAD["by_language"]
    assert body["total"] == 4
    assert body["languages"] == ["eng", "lug", "swa"]


def test_get_speakers_by_language_ok(speakers_handler):
    app = build_test_app(speakers_handler)
    client = TestClient(app)
    r = client.get("/speakers/lug")
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "lug"
    assert body["speakers"] == ["salt_lug_0001", "salt_lug_0002"]
    assert body["count"] == 2


def test_get_speakers_by_language_unknown_returns_404(speakers_handler):
    app = build_test_app(speakers_handler)
    client = TestClient(app)
    r = client.get("/speakers/xyz")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "unknown_language"
    assert "xyz" in body["detail"]
