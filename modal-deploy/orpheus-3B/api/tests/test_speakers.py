import asyncio

import pytest


class _FakeModal:
    def __init__(self, payload, *, fail_first: bool = False):
        self.payload = payload
        self.calls = 0
        self.fail_first = fail_first

    async def speakers(self) -> dict:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("modal down")
        return self.payload


_PAYLOAD = {
    "default": "salt_lug_0001",
    "by_language": {
        "lug": ["salt_lug_0001", "salt_lug_0002"],
        "eng": ["salt_eng_0001"],
    },
}


@pytest.mark.asyncio
async def test_cache_loads_and_returns_catalog():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    cat = await cache.get()
    assert cat.default == "salt_lug_0001"
    assert "lug" in cat.by_language
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_cache_reuses_within_ttl():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.get()
    await cache.get()
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_cache_refreshes_after_ttl():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=0)  # always expired
    await cache.get()
    await cache.get()
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_cache_fail_open_when_modal_down_on_startup():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD, fail_first=True)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    # try_warm() must not raise; should leave cache empty
    await cache.try_warm()
    assert cache.is_warm is False


@pytest.mark.asyncio
async def test_validate_known_speaker_passes():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()
    await cache.validate_speaker("salt_lug_0001", language=None)
    await cache.validate_speaker("salt_lug_0001", language="lug")
    # also returns the resolved language
    lang = await cache.language_for("salt_lug_0001")
    assert lang == "lug"


@pytest.mark.asyncio
async def test_validate_unknown_speaker_raises():
    from api.errors import InvalidSpeakerError
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()
    with pytest.raises(InvalidSpeakerError):
        await cache.validate_speaker("salt_zzz_9999", language=None)


@pytest.mark.asyncio
async def test_validate_unknown_language_raises():
    from api.errors import UnknownLanguageError
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()
    with pytest.raises(UnknownLanguageError):
        await cache.validate_speaker("salt_lug_0001", language="zzz")


@pytest.mark.asyncio
async def test_validate_speaker_for_language_mismatch_raises():
    from api.errors import InvalidSpeakerForLanguageError
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()
    with pytest.raises(InvalidSpeakerForLanguageError):
        await cache.validate_speaker("salt_lug_0001", language="eng")


@pytest.mark.asyncio
async def test_validate_falls_open_when_not_warm():
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD, fail_first=True)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()  # leaves cache unwarmed
    assert cache.is_warm is False
    # validate_speaker must NOT raise when cache is unwarmed (fail-open)
    await cache.validate_speaker("anything", language=None)
    await cache.validate_speaker("anything", language="anylang")


@pytest.mark.asyncio
async def test_language_for_does_not_refresh_or_raise():
    """language_for must read self._catalog directly without refreshing.

    Even if the TTL has expired and Modal would now fail, language_for
    must return the cached value (or None) without raising.
    """
    from api.speakers import SpeakersCache

    fake = _FakeModal(_PAYLOAD)
    cache = SpeakersCache(modal=fake, ttl_seconds=60)
    await cache.try_warm()
    assert fake.calls == 1

    # Make subsequent modal.speakers() raise. If language_for tried to
    # refresh, this would surface; we want None or a stale-cache hit
    # instead.
    async def boom():
        raise RuntimeError("modal exploded")

    fake.speakers = boom  # type: ignore[assignment]

    lang = await cache.language_for("salt_lug_0001")
    assert lang == "lug"
    # No additional modal calls were made.
    assert fake.calls == 1

    # Unknown speaker returns None, not a raise.
    lang_unknown = await cache.language_for("salt_zzz_9999")
    assert lang_unknown is None
