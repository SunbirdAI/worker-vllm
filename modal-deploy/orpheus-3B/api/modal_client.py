"""Async httpx wrapper for the Modal-deployed Orpheus inference app.

Owns timeout & retry policy. Exposes:
- health()
- speakers()
- tts(...) -> TTSAudio
- tts_batch([...]) -> list[TTSAudio]

Retry policy: one app-level retry on (httpx.ReadTimeout, 5xx) with a
backoff of `retry_backoff_seconds + jitter`. After exhaustion, the right
APIError subclass is raised so the route layer doesn't have to know the
upstream details.
"""

import asyncio
import base64
import binascii
import logging
import random
from dataclasses import dataclass

import httpx

from api.errors import UpstreamTimeoutError, UpstreamUnavailableError

logger = logging.getLogger("orpheus_api.modal")

_RETRY_STATUSES = {502, 503, 504}


@dataclass
class TTSAudio:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    speaker_id: str
    chunks: int | None = None


class ModalClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.client = client
        self.backoff = retry_backoff_seconds

    async def health(self) -> bool:
        try:
            r = await self.client.get("/health", timeout=10.0)
        except httpx.HTTPError as exc:
            logger.warning("modal_health_unreachable: %s", exc)
            return False
        return r.is_success

    async def speakers(self) -> dict:
        return await self._json_get("/speakers")

    async def tts(
        self,
        *,
        text: str,
        speaker_id: str,
        seed: int | None,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> TTSAudio:
        body = {
            "text": text,
            "speaker_id": speaker_id,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }
        resp = await self._post_with_retry("/tts", json=body)
        sr = int(resp.headers.get("X-Sample-Rate", "24000"))
        dur = float(resp.headers.get("X-Duration-Seconds", "0"))
        chunks_hdr = resp.headers.get("X-Chunks")
        try:
            chunks = int(chunks_hdr) if chunks_hdr is not None else None
        except ValueError:
            chunks = None
        return TTSAudio(
            audio_bytes=resp.content,
            sample_rate=sr,
            duration_seconds=dur,
            speaker_id=resp.headers.get("X-Speaker-Id", speaker_id),
            chunks=chunks,
        )

    async def tts_batch(self, items: list[dict]) -> list[TTSAudio]:
        body = {"items": items}
        resp = await self._post_with_retry("/tts/batch", json=body)
        try:
            data = resp.json()
        except ValueError as exc:
            raise UpstreamUnavailableError(
                f"non-JSON response from /tts/batch: {exc}"
            ) from exc
        results = []
        for idx, r in enumerate(data.get("results", [])):
            try:
                wav = base64.b64decode(r["audio_wav_b64"], validate=True)
            except (KeyError, ValueError, binascii.Error) as exc:
                raise UpstreamUnavailableError(
                    f"malformed result at index {idx}: {exc}"
                ) from exc
            results.append(
                TTSAudio(
                    audio_bytes=wav,
                    sample_rate=int(r.get("sample_rate", 24000)),
                    duration_seconds=float(r.get("duration_sec", 0.0)),
                    speaker_id=r.get("speaker_id", ""),
                )
            )
        return results

    # ----- internals -----

    async def _json_get(self, path: str) -> dict:
        resp = await self._with_retry(lambda: self.client.get(path))
        return resp.json()

    async def _post_with_retry(self, path: str, *, json: dict) -> httpx.Response:
        return await self._with_retry(lambda: self.client.post(path, json=json))

    async def _with_retry(self, op):
        for attempt in (1, 2):
            try:
                resp = await op()
            except httpx.ReadTimeout as exc:
                if attempt == 1:
                    await self._sleep_backoff()
                    continue
                raise UpstreamTimeoutError(
                    f"Modal request timed out after {attempt} attempt(s)"
                ) from exc
            except httpx.HTTPError as exc:
                if attempt == 1:
                    await self._sleep_backoff()
                    continue
                raise UpstreamUnavailableError(
                    f"Modal request failed: {exc}"
                ) from exc

            if resp.status_code in _RETRY_STATUSES and attempt == 1:
                await self._sleep_backoff()
                continue
            if resp.is_success:
                return resp
            # 4xx (after our pre-flight checks) → bug; surface as 502.
            # Includes Modal's 422 text_too_long from the chunker cap —
            # currently unreachable because the gateway's TTSRequest enforces
            # max_length=2000 on `text` (≈10 chunks at the 220-char target,
            # well under MAX_CHUNKS_PER_REQUEST=48). Decouple cautiously.
            raise UpstreamUnavailableError(
                f"Modal returned {resp.status_code}: {resp.text[:200]}"
            )

        # Unreachable: the loop above always returns on success or raises on
        # the second failed attempt.
        raise AssertionError("unreachable")

    async def _sleep_backoff(self) -> None:
        if self.backoff <= 0:
            return
        await asyncio.sleep(self.backoff + random.uniform(0, self.backoff / 2))
