import base64
import json
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from api.tests.conftest import SPEAKERS_PAYLOAD, build_test_app, make_wav


def _modal_handler(
    *, audio: bytes, duration: float = 1.5, x_chunks: str | None = None
):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/tts":
            headers = {
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "24000",
                "X-Duration-Seconds": str(duration),
                "X-Speaker-Id": json.loads(req.content)["speaker_id"],
            }
            if x_chunks is not None:
                headers["X-Chunks"] = x_chunks
            return httpx.Response(200, content=audio, headers=headers)
        return httpx.Response(404)

    return handler


def test_tts_happy_path():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav, duration=2.0))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "Mwattu", "speaker_id": "salt_lug_0001"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["speaker_id"] == "salt_lug_0001"
    assert body["language"] == "lug"
    assert body["sample_rate"] == 24000
    assert body["duration_seconds"] == pytest.approx(2.0)
    assert body["audio_size_bytes"] == len(wav)
    assert body["gcs_object"].startswith("tts/")
    assert body["audio_url"].startswith("https://signed.example/")
    t = body["timings_ms"]
    assert t["inference_ms"] >= 0
    assert t["upload_ms"] >= 0
    assert t["total_ms"] >= 0
    assert "request_id" in body


def test_tts_response_includes_chunks_when_upstream_sets_header():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav, duration=2.0, x_chunks="4"))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "long text", "speaker_id": "salt_lug_0001"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks"] == 4


def test_tts_response_chunks_is_null_when_upstream_omits_header():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav, duration=1.0))  # no x_chunks
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "short", "speaker_id": "salt_lug_0001"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks"] is None


def test_tts_language_field_validated_ok():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "hi", "speaker_id": "salt_lug_0001", "language": "lug"},
    )
    assert r.status_code == 200, r.text


def test_tts_unknown_speaker_400():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "hi", "speaker_id": "salt_zzz_9999"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_speaker"


def test_tts_unknown_language_400():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "hi", "speaker_id": "salt_lug_0001", "language": "xyz"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "unknown_language"


def test_tts_speaker_for_language_mismatch_400():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "hi", "speaker_id": "salt_lug_0001", "language": "eng"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_speaker_for_language"


def test_tts_modal_502_returns_502():
    def handler(req):
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(502, text="bad gateway")

    app = build_test_app(handler)
    client = TestClient(app)
    r = client.post("/tts", json={"text": "hi", "speaker_id": "salt_lug_0001"})
    assert r.status_code == 502
    assert r.json()["error"] == "upstream_unavailable"


def test_tts_modal_422_text_too_long_surfaces_as_502():
    # Modal raises 422 text_too_long when the chunker exceeds
    # MAX_CHUNKS_PER_REQUEST=48. The gateway's TTSRequest.max_length=2000
    # makes this practically unreachable, but the cross-layer mapping
    # (Modal 4xx → gateway 502 upstream_unavailable via _with_retry's
    # catch-all) is worth pinning down by test in case either limit moves.
    def handler(req):
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            422,
            json={
                "detail": {
                    "error": "text_too_long",
                    "detail": "text produced 60 chunks (limit 48)",
                }
            },
        )

    app = build_test_app(handler)
    client = TestClient(app)
    r = client.post("/tts", json={"text": "hi", "speaker_id": "salt_lug_0001"})
    assert r.status_code == 502
    assert r.json()["error"] == "upstream_unavailable"


def test_tts_request_id_echoed():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post(
        "/tts",
        json={"text": "hi", "speaker_id": "salt_lug_0001"},
        headers={"X-Request-ID": "client-rid"},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "client-rid"
    assert r.json()["request_id"] == "client-rid"


def test_tts_validation_error_422_on_empty_text():
    wav = make_wav()
    app = build_test_app(_modal_handler(audio=wav))
    client = TestClient(app)
    r = client.post("/tts", json={"text": "", "speaker_id": "salt_lug_0001"})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"
