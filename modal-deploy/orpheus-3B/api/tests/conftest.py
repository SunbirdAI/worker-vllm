"""Shared pytest fixtures: TestClient with mocked Modal + fake GCS."""

import base64
import struct
from typing import Callable

import httpx
import pytest
from fastapi.testclient import TestClient


def make_wav(n_samples: int = 2400) -> bytes:
    sample_rate = 24000
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    data = b"\x00\x00" * n_samples
    riff = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    fmt = b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits
    )
    dat = b"data" + struct.pack("<I", len(data)) + data
    return riff + fmt + dat


SPEAKERS_PAYLOAD = {
    "default": "salt_lug_0001",
    "by_language": {
        "lug": ["salt_lug_0001", "salt_lug_0002"],
        "eng": ["salt_eng_0001"],
        "swa": ["waxal_swa_0006"],
    },
}


class FakeBlob:
    def __init__(self, name, bucket):
        self.name = name
        self.bucket = bucket
        self.uploaded = None
        self.metadata = {}

    def upload_from_string(self, data, content_type, timeout=None):
        self.uploaded = data
        self.bucket.blobs[self.name] = self

    def generate_signed_url(self, **kwargs):
        return f"https://signed.example/{self.bucket.name}/{self.name}"


class FakeBucket:
    def __init__(self, name):
        self.name = name
        self.blobs = {}
        self._exists = True

    def blob(self, name):
        return FakeBlob(name, self)

    def exists(self):
        return self._exists


class FakeGCSClient:
    def __init__(self):
        self.bucket_obj = FakeBucket("test-bucket")

    def bucket(self, name):
        return self.bucket_obj


def build_test_app(handler: Callable, *, warm_speakers: bool = True):
    """Build the FastAPI app with httpx MockTransport + FakeGCSClient,
    bypassing the real lifespan."""
    from api.errors import register_exception_handlers, request_id_middleware
    from api.modal_client import ModalClient
    from api.routes import router
    from api.speakers import SpeakersCache
    from api.storage import StorageBackend
    from fastapi import FastAPI

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="https://modal.test")
    modal = ModalClient(client=http, retry_backoff_seconds=0.0)
    speakers = SpeakersCache(modal=modal, ttl_seconds=60)
    storage = StorageBackend(
        client=FakeGCSClient(),
        bucket_name="test-bucket",
        object_prefix="tts",
        signed_url_expiry_minutes=30,
    )

    app = FastAPI()
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(router)
    app.state.http = http
    app.state.modal = modal
    app.state.speakers = speakers
    app.state.storage = storage

    # Minimal settings stand-in (real Settings requires env vars).
    class _TestSettings:
        max_batch_size = 16
    app.state.settings = _TestSettings()

    if warm_speakers:
        # Eagerly warm via the same MockTransport so the speakers cache is
        # populated before TestClient starts its own event loop. The cached
        # Catalog is a plain dataclass, so it's safe to carry across loops.
        import asyncio

        asyncio.run(speakers.try_warm())

    return app


@pytest.fixture
def speakers_handler():
    """Returns an httpx.MockTransport handler that serves /speakers."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/speakers":
            return httpx.Response(200, json=SPEAKERS_PAYLOAD)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    return handler
