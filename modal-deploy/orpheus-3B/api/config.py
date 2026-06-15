"""Environment-driven settings for the Orpheus TTS FastAPI service.

Reads required vars from the environment; missing required → startup fails
with a clear message. Optional vars have sensible defaults documented in
.env.example.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required
    orpheus_modal_url: str = Field(..., description="Base URL of the Modal app")
    gcs_bucket_name: str = Field(..., description="Existing GCS bucket name")
    google_application_credentials: str = Field(
        ..., description="Path to service account JSON key file"
    )

    # GCS
    gcs_object_prefix: str = Field("tts")
    gcs_signed_url_expiry_minutes: int = Field(30, ge=1, le=7 * 24 * 60)

    # Modal client
    modal_request_timeout_seconds: float = Field(180.0, gt=0)
    modal_connect_timeout_seconds: float = Field(10.0, gt=0)
    modal_retry_backoff_seconds: float = Field(0.5, ge=0)

    # Batching / caching
    max_batch_size: int = Field(16, ge=1, le=128)
    speakers_cache_ttl_seconds: int = Field(60, ge=1)

    # Logging / serving
    log_level: str = Field("INFO")
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)

    @field_validator("google_application_credentials")
    @classmethod
    def _creds_file_exists(cls, v: str) -> str:
        p = Path(v)
        if not p.is_file():
            raise ValueError(
                f"google_application_credentials file not found: {v}"
            )
        return v

    @field_validator("orpheus_modal_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor — Settings is instantiated on first call and reused.

    Tests should instantiate Settings() directly with monkeypatched env vars
    rather than going through this function, so the cache doesn't leak state
    across tests.
    """
    return Settings()  # type: ignore[call-arg]
