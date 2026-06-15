"""Pydantic v2 request/response models for the Orpheus TTS API."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, computed_field

# ----- TTS request -----

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize.")
    speaker_id: str = Field(
        "salt_lug_0001",
        description="Speaker tag from the finetune set (see GET /speakers).",
    )
    language: Optional[str] = Field(
        None,
        description=(
            "Optional ISO 639-3 language code (e.g. 'lug', 'eng'). "
            "If set, speaker_id must belong to it."
        ),
    )
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility.")
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    max_tokens: int = Field(1200, ge=64, le=4096)


# ----- Batch (size is also enforced at the route layer against config.max_batch_size) -----

class TTSBatchRequest(BaseModel):
    items: list[TTSRequest] = Field(..., min_length=1, max_length=128)


# ----- Timings -----

class Timings(BaseModel):
    inference_ms: float
    upload_ms: float
    signed_url_ms: float
    total_ms: float


class BatchTimings(BaseModel):
    inference_ms: float
    upload_ms: float
    total_ms: float


# ----- TTS response (single) -----

class TTSResponse(BaseModel):
    audio_url: HttpUrl
    audio_url_expires_at: datetime
    speaker_id: str
    language: Optional[str] = None
    sample_rate: int = 24000
    duration_seconds: float
    chunks: Optional[int] = None
    audio_size_bytes: int
    gcs_object: str
    request_id: str
    timings_ms: Timings


# ----- Batch results -----

class TTSBatchItemResult(BaseModel):
    index: int
    status: Literal["ok", "error"]
    speaker_id: str

    # success-only
    audio_url: Optional[HttpUrl] = None
    audio_url_expires_at: Optional[datetime] = None
    language: Optional[str] = None
    sample_rate: int = 24000
    duration_seconds: Optional[float] = None
    audio_size_bytes: Optional[int] = None
    gcs_object: Optional[str] = None
    request_id: Optional[str] = None

    # error-only
    error_code: Optional[str] = None
    error_detail: Optional[str] = None


class TTSBatchResponse(BaseModel):
    results: list[TTSBatchItemResult]
    timings_ms: BatchTimings
    request_id: str


# ----- Catalog -----

class SpeakersResponse(BaseModel):
    default: str
    by_language: dict[str, list[str]]

    @computed_field
    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_language.values())

    @computed_field
    @property
    def languages(self) -> list[str]:
        return sorted(self.by_language.keys())


class LanguageSpeakersResponse(BaseModel):
    language: str
    speakers: list[str]

    @computed_field
    @property
    def count(self) -> int:
        return len(self.speakers)


# ----- Health -----

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["orpheus-tts-api"] = "orpheus-tts-api"


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    upstreams: dict[str, str]  # {"modal": "ok"|"<error>", "gcs": "ok"|"<error>"}


# ----- Errors -----

class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
