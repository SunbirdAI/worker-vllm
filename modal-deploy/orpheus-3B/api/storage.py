"""GCS upload + v4 signed URL helpers.

google-cloud-storage is sync. We wrap each blocking call in
asyncio.to_thread so the FastAPI event loop is free during uploads —
this is what lets `/tts/batch` upload in parallel via asyncio.gather.
"""

import asyncio
import datetime as dt
import logging
import time
import uuid
from dataclasses import dataclass

from api.errors import StorageUnavailableError

logger = logging.getLogger("orpheus_api.storage")


@dataclass
class UploadResult:
    gcs_object: str
    audio_url: str
    audio_url_expires_at: dt.datetime
    audio_size_bytes: int
    upload_ms: float
    signed_url_ms: float


class StorageBackend:
    def __init__(
        self,
        *,
        client,
        bucket_name: str,
        object_prefix: str,
        signed_url_expiry_minutes: int,
    ) -> None:
        self._client = client
        self.bucket_name = bucket_name
        self.object_prefix = object_prefix.strip("/")
        self.signed_expiry = dt.timedelta(minutes=signed_url_expiry_minutes)

    def _object_name(self) -> str:
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        return f"{self.object_prefix}/{date}/{uuid.uuid4().hex}.wav"

    async def upload_wav(
        self, *, audio_bytes: bytes, content_type: str = "audio/wav"
    ) -> UploadResult:
        name = self._object_name()
        try:
            return await asyncio.to_thread(
                self._upload_and_sign, name, audio_bytes, content_type
            )
        except StorageUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StorageUnavailableError(f"GCS failure: {exc}") from exc

    def _upload_and_sign(
        self, name: str, audio_bytes: bytes, content_type: str
    ) -> UploadResult:
        bucket = self._client.bucket(self.bucket_name)
        blob = bucket.blob(name)
        # Hint to any CDN/proxy not to cache the (short-lived) signed-URL
        # response. The metadata setter on real GCS blobs accepts a dict;
        # guard narrowly against the rare case where a custom blob impl
        # exposes a non-writable property.
        try:
            blob.metadata = {"Cache-Control": "private, max-age=0"}
        except (AttributeError, TypeError) as exc:
            logger.debug("blob_metadata_unset: %s", exc)

        t0 = _monotonic_ms()
        try:
            blob.upload_from_string(
                audio_bytes, content_type=content_type, timeout=60
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageUnavailableError(f"upload failed: {exc}") from exc
        t1 = _monotonic_ms()

        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=self.signed_expiry,
                method="GET",
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageUnavailableError(f"sign failed: {exc}") from exc
        t2 = _monotonic_ms()

        return UploadResult(
            gcs_object=name,
            audio_url=url,
            audio_url_expires_at=dt.datetime.now(dt.timezone.utc) + self.signed_expiry,
            audio_size_bytes=len(audio_bytes),
            upload_ms=t1 - t0,
            signed_url_ms=t2 - t1,
        )

    async def check_bucket(self) -> bool:
        try:
            return await asyncio.to_thread(self._check_bucket_blocking)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gcs_check_failed: %s", exc)
            return False

    def _check_bucket_blocking(self) -> bool:
        return bool(self._client.bucket(self.bucket_name).exists())


def _monotonic_ms() -> float:
    return time.monotonic() * 1000.0
