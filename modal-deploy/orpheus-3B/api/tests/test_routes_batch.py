import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api.tests.conftest import SPEAKERS_PAYLOAD, build_test_app, make_wav


def _batch_handler(*, audios: list[bytes], status: int = 200):
    """Return /tts/batch JSON payload with per-item audios."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if req.url.path == "/tts/batch":
            if status != 200:
                return httpx.Response(status, text="upstream error")
            body = json.loads(req.content)
            assert len(audios) == len(body["items"])
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "text": it["text"],
                            "speaker_id": it["speaker_id"],
                            "sample_rate": 24000,
                            "duration_sec": 1.0,
                            "audio_wav_b64": base64.b64encode(a).decode("ascii"),
                        }
                        for it, a in zip(body["items"], audios)
                    ]
                },
            )
        return httpx.Response(404)

    return handler


def test_batch_happy_path():
    wavs = [make_wav(1000), make_wav(2000), make_wav(3000)]
    app = build_test_app(_batch_handler(audios=wavs))
    client = TestClient(app)
    r = client.post(
        "/tts/batch",
        json={
            "items": [
                {"text": "a", "speaker_id": "salt_lug_0001"},
                {"text": "b", "speaker_id": "salt_eng_0001"},
                {"text": "c", "speaker_id": "waxal_swa_0006"},
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 3
    for i, res in enumerate(body["results"]):
        assert res["index"] == i
        assert res["status"] == "ok"
        assert res["audio_size_bytes"] == len(wavs[i])
        assert res["audio_url"].startswith("https://signed.example/")
        assert res["gcs_object"].startswith("tts/")
        assert res["sample_rate"] == 24000
    assert body["timings_ms"]["inference_ms"] >= 0
    assert body["timings_ms"]["upload_ms"] >= 0


def test_batch_rejects_empty():
    app = build_test_app(_batch_handler(audios=[]))
    client = TestClient(app)
    r = client.post("/tts/batch", json={"items": []})
    assert r.status_code == 422


def test_batch_rejects_unknown_speaker_in_any_item():
    wavs = [make_wav()]
    app = build_test_app(_batch_handler(audios=wavs))
    client = TestClient(app)
    r = client.post(
        "/tts/batch",
        json={
            "items": [
                {"text": "a", "speaker_id": "salt_lug_0001"},
                {"text": "b", "speaker_id": "salt_zzz_9999"},
            ]
        },
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_speaker"
    assert "1" in body["detail"]


def test_batch_all_modal_502_returns_502():
    app = build_test_app(_batch_handler(audios=[], status=502))
    client = TestClient(app)
    r = client.post(
        "/tts/batch",
        json={"items": [{"text": "a", "speaker_id": "salt_lug_0001"}]},
    )
    assert r.status_code == 502


def test_batch_per_item_gcs_failure_does_not_sink_others(monkeypatch):
    """If GCS upload for one item raises, the other items still return OK."""
    wavs = [make_wav(1000), make_wav(2000)]
    app = build_test_app(_batch_handler(audios=wavs))

    original = app.state.storage.upload_wav
    calls = {"n": 0}

    async def flaky_upload(*, audio_bytes, content_type):
        calls["n"] += 1
        if calls["n"] == 2:
            from api.errors import StorageUnavailableError
            raise StorageUnavailableError("fake gcs failure on item 2")
        return await original(audio_bytes=audio_bytes, content_type=content_type)

    app.state.storage.upload_wav = flaky_upload  # type: ignore[assignment]

    client = TestClient(app)
    r = client.post(
        "/tts/batch",
        json={
            "items": [
                {"text": "a", "speaker_id": "salt_lug_0001"},
                {"text": "b", "speaker_id": "salt_eng_0001"},
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    statuses = [r["status"] for r in body["results"]]
    assert statuses.count("ok") == 1
    assert statuses.count("error") == 1
    failed = [r for r in body["results"] if r["status"] == "error"][0]
    assert failed["error_code"] == "storage_unavailable"
