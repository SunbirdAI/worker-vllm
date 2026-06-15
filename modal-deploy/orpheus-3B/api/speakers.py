"""TTL-cached speakers catalog + per-request validation helpers.

The catalog is fetched lazily on first call and refreshed when older than
ttl_seconds. If the upstream fetch fails, the cache stays unwarmed and
validation falls open (requests proceed; unknown-speaker errors then
surface from Modal as 502).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from api.errors import (
    InvalidSpeakerError,
    InvalidSpeakerForLanguageError,
    UnknownLanguageError,
)

logger = logging.getLogger("orpheus_api.speakers")


@dataclass
class Catalog:
    default: str
    by_language: dict[str, list[str]]
    speaker_to_language: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict) -> "Catalog":
        by_lang = payload.get("by_language", {}) or {}
        s2l: dict[str, str] = {}
        for lang, speakers in by_lang.items():
            for sp in speakers:
                s2l[sp] = lang
        return cls(
            default=payload.get("default", ""),
            by_language=by_lang,
            speaker_to_language=s2l,
        )


class SpeakersCache:
    def __init__(self, modal, ttl_seconds: int) -> None:
        self.modal = modal
        self.ttl = ttl_seconds
        self._catalog: Catalog | None = None
        self._loaded_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def is_warm(self) -> bool:
        return self._catalog is not None

    async def try_warm(self) -> None:
        """Attempt initial load. Never raises — failure leaves cache cold."""
        try:
            await self._refresh()
        except Exception as exc:  # noqa: BLE001
            logger.warning("speakers_cache_warm_failed: %s", exc)

    async def get(self) -> Catalog:
        if self._catalog is None or (time.monotonic() - self._loaded_at) > self.ttl:
            await self._refresh()
        assert self._catalog is not None  # _refresh raises on failure
        return self._catalog

    async def _refresh(self) -> None:
        async with self._lock:
            # Double-check after acquiring lock.
            if (
                self._catalog is not None
                and (time.monotonic() - self._loaded_at) <= self.ttl
            ):
                return
            payload = await self.modal.speakers()
            self._catalog = Catalog.from_payload(payload)
            self._loaded_at = time.monotonic()

    async def language_for(self, speaker_id: str) -> str | None:
        """Reverse lookup: speaker_id -> language.

        Reads the cached catalog directly without triggering a refresh.
        Returns None if the cache is cold or the speaker is unknown.
        Never raises — callers use this in success paths after TTS
        generation has already completed, so a transient Modal blip must
        not fail the response.
        """
        if self._catalog is None:
            return None
        return self._catalog.speaker_to_language.get(speaker_id)

    async def validate_speaker(
        self, speaker_id: str, *, language: str | None
    ) -> None:
        """Raises 400-class APIError if invalid. Falls open if cache unwarmed."""
        if not self.is_warm:
            return  # fail-open
        cat = await self.get()
        if speaker_id not in cat.speaker_to_language:
            raise InvalidSpeakerError(
                f"speaker_id '{speaker_id}' not found; see /speakers"
            )
        if language is not None:
            if language not in cat.by_language:
                raise UnknownLanguageError(
                    f"language '{language}' not supported; "
                    f"supported: {sorted(cat.by_language)}"
                )
            actual = cat.speaker_to_language[speaker_id]
            if actual != language:
                raise InvalidSpeakerForLanguageError(
                    f"speaker '{speaker_id}' is for language '{actual}', "
                    f"not '{language}'; see /speakers/{language}"
                )
