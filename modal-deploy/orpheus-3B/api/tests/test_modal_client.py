import asyncio
import base64
import io
import json
import struct
from typing import Callable

import httpx
import pytest

from api.errors import UpstreamTimeoutError, UpstreamUnavailableError
from api.modal_client import ModalClient


def _wav_bytes(n_samples: int = 2400) -> bytes:
    """Minimal mono 16-bit 24 kHz WAV with `n_samples` of silence."""
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


def _make_client(handler: Callable):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://modal.test")


@pytest.mark.asyncio
async def test_health_ok():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        ok = await mc.health()
        assert ok is True


@pytest.mark.asyncio
async def test_speakers_returns_payload():
    payload = {
        "default": "salt_lug_0001",
        "by_language": {"lug": ["salt_lug_0001"], "eng": ["salt_eng_0001"]},
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/speakers"
        return httpx.Response(200, json=payload)

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        out = await mc.speakers()
        assert out == payload


@pytest.mark.asyncio
async def test_tts_returns_wav_and_duration():
    wav = _wav_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/tts"
        body = json.loads(req.content)
        assert body["text"] == "hi"
        assert body["speaker_id"] == "salt_lug_0001"
        return httpx.Response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "24000",
                "X-Duration-Seconds": "1.234",
                "X-Speaker-Id": "salt_lug_0001",
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="hi",
            speaker_id="salt_lug_0001",
            seed=None,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200,
        )
        assert result.audio_bytes == wav
        assert result.duration_seconds == pytest.approx(1.234)
        assert result.sample_rate == 24000


@pytest.mark.asyncio
async def test_tts_parses_x_chunks_header():
    wav = _wav_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "24000",
                "X-Duration-Seconds": "5.0",
                "X-Speaker-Id": "salt_lug_0001",
                "X-Chunks": "6",
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="long",
            speaker_id="salt_lug_0001",
            seed=None,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200,
        )
        assert result.chunks == 6


@pytest.mark.asyncio
async def test_tts_chunks_defaults_to_none_when_header_absent():
    wav = _wav_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "24000",
                "X-Duration-Seconds": "1.0",
                # no X-Chunks header (simulates older Modal deploy)
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="hi",
            speaker_id="salt_lug_0001",
            seed=None,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200,
        )
        assert result.chunks is None


@pytest.mark.asyncio
async def test_tts_chunks_is_none_when_header_is_not_an_integer():
    wav = _wav_bytes()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=wav,
            headers={
                "Content-Type": "audio/wav",
                "X-Sample-Rate": "24000",
                "X-Duration-Seconds": "1.0",
                "X-Chunks": "abc",  # malformed; simulates upstream bug
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="hi",
            speaker_id="salt_lug_0001",
            seed=None,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200,
        )
        assert result.chunks is None


@pytest.mark.asyncio
async def test_tts_retries_once_on_502_then_succeeds():
    wav = _wav_bytes()
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(
            200, content=wav, headers={"X-Duration-Seconds": "0.5"}
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="hi",
            speaker_id="salt_lug_0001",
            seed=None,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200,
        )
        assert result.audio_bytes == wav
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_tts_persistent_502_raises_upstream_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        with pytest.raises(UpstreamUnavailableError):
            await mc.tts(
                text="hi", speaker_id="salt_lug_0001",
                seed=None, temperature=0.6, top_p=0.95,
                repetition_penalty=1.1, max_tokens=1200,
            )


@pytest.mark.asyncio
async def test_tts_read_timeout_then_success():
    wav = _wav_bytes()
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(
            200, content=wav, headers={"X-Duration-Seconds": "0.5"}
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        result = await mc.tts(
            text="hi", speaker_id="salt_lug_0001",
            seed=None, temperature=0.6, top_p=0.95,
            repetition_penalty=1.1, max_tokens=1200,
        )
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_tts_persistent_timeout_raises_upstream_timeout():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        with pytest.raises(UpstreamTimeoutError):
            await mc.tts(
                text="hi", speaker_id="salt_lug_0001",
                seed=None, temperature=0.6, top_p=0.95,
                repetition_penalty=1.1, max_tokens=1200,
            )


@pytest.mark.asyncio
async def test_tts_batch_parses_b64_results():
    wav1 = _wav_bytes(1200)
    wav2 = _wav_bytes(2400)
    payload = {
        "results": [
            {
                "text": "a",
                "speaker_id": "salt_lug_0001",
                "sample_rate": 24000,
                "duration_sec": 0.05,
                "audio_wav_b64": base64.b64encode(wav1).decode("ascii"),
            },
            {
                "text": "b",
                "speaker_id": "salt_eng_0001",
                "sample_rate": 24000,
                "duration_sec": 0.10,
                "audio_wav_b64": base64.b64encode(wav2).decode("ascii"),
            },
        ]
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/tts/batch"
        return httpx.Response(200, json=payload)

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        items = [
            {"text": "a", "speaker_id": "salt_lug_0001", "seed": None,
             "temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1, "max_tokens": 1200},
            {"text": "b", "speaker_id": "salt_eng_0001", "seed": None,
             "temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.1, "max_tokens": 1200},
        ]
        out = await mc.tts_batch(items)
        assert len(out) == 2
        assert out[0].audio_bytes == wav1
        assert out[0].duration_seconds == pytest.approx(0.05)
        assert out[1].audio_bytes == wav2


@pytest.mark.asyncio
async def test_tts_batch_malformed_base64_raises_upstream_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "text": "a",
                        "speaker_id": "salt_lug_0001",
                        "sample_rate": 24000,
                        "duration_sec": 0.05,
                        "audio_wav_b64": "this is not valid base64!@#$",
                    }
                ]
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        with pytest.raises(UpstreamUnavailableError):
            await mc.tts_batch(
                [
                    {
                        "text": "a", "speaker_id": "salt_lug_0001",
                        "seed": None, "temperature": 0.6, "top_p": 0.95,
                        "repetition_penalty": 1.1, "max_tokens": 1200,
                    }
                ]
            )


@pytest.mark.asyncio
async def test_tts_batch_missing_audio_field_raises_upstream_unavailable():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "text": "a",
                        "speaker_id": "salt_lug_0001",
                        "sample_rate": 24000,
                        "duration_sec": 0.05,
                        # audio_wav_b64 missing entirely
                    }
                ]
            },
        )

    async with _make_client(handler) as ac:
        mc = ModalClient(client=ac, retry_backoff_seconds=0.0)
        with pytest.raises(UpstreamUnavailableError):
            await mc.tts_batch(
                [
                    {
                        "text": "a", "speaker_id": "salt_lug_0001",
                        "seed": None, "temperature": 0.6, "top_p": 0.95,
                        "repetition_penalty": 1.1, "max_tokens": 1200,
                    }
                ]
            )
