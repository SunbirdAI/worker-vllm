from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


def test_tts_request_defaults():
    from api.models import TTSRequest
    r = TTSRequest(text="hello")
    assert r.speaker_id == "salt_lug_0001"
    assert r.language is None
    assert r.temperature == 0.6
    assert r.top_p == 0.95
    assert r.repetition_penalty == 1.1
    assert r.max_tokens == 1200
    assert r.seed is None


def test_tts_request_text_required():
    from api.models import TTSRequest
    with pytest.raises(ValidationError):
        TTSRequest(text="")


def test_tts_request_text_too_long():
    from api.models import TTSRequest
    with pytest.raises(ValidationError):
        TTSRequest(text="x" * 2001)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("repetition_penalty", 0.99),
        ("repetition_penalty", 2.01),
        ("max_tokens", 63),
        ("max_tokens", 4097),
    ],
)
def test_tts_request_bounds(field, value):
    from api.models import TTSRequest
    with pytest.raises(ValidationError):
        TTSRequest(text="ok", **{field: value})


def test_tts_batch_request_size_bounds():
    from api.models import TTSBatchRequest, TTSRequest
    with pytest.raises(ValidationError):
        TTSBatchRequest(items=[])
    items = [TTSRequest(text=f"t{i}") for i in range(129)]
    with pytest.raises(ValidationError):
        TTSBatchRequest(items=items)


def test_timings_total():
    from api.models import Timings
    t = Timings(inference_ms=100.0, upload_ms=50.0, signed_url_ms=10.0, total_ms=160.0)
    assert t.total_ms == 160.0


def test_tts_response_ok():
    from api.models import Timings, TTSResponse
    resp = TTSResponse(
        audio_url="https://storage.googleapis.com/x?token=y",
        audio_url_expires_at=datetime.now(timezone.utc),
        speaker_id="salt_lug_0001",
        language="lug",
        sample_rate=24000,
        duration_seconds=2.4,
        audio_size_bytes=115_244,
        gcs_object="tts/2026-05-12/abc.wav",
        request_id="abc-123",
        timings_ms=Timings(inference_ms=1000, upload_ms=200, signed_url_ms=10, total_ms=1210),
    )
    assert str(resp.audio_url).startswith("https://")


def test_speakers_response_total_computed():
    from api.models import SpeakersResponse
    r = SpeakersResponse(
        default="salt_lug_0001",
        by_language={"lug": ["salt_lug_0001", "salt_lug_0002"], "eng": ["salt_eng_0001"]},
    )
    assert r.total == 3
    assert r.languages == ["eng", "lug"]


def test_language_speakers_response():
    from api.models import LanguageSpeakersResponse
    r = LanguageSpeakersResponse(language="lug", speakers=["salt_lug_0001"])
    assert r.count == 1


def test_error_response_shape():
    from api.models import ErrorResponse
    e = ErrorResponse(error="invalid_speaker", detail="not found", request_id="abc")
    assert e.error == "invalid_speaker"
