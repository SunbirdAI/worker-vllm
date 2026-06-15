import datetime as dt

import pytest


class _FakeBlob:
    def __init__(self, name: str, bucket: "_FakeBucket"):
        self.name = name
        self.bucket = bucket
        self.uploaded_bytes: bytes | None = None
        self.uploaded_content_type: str | None = None
        self.uploaded_metadata: dict | None = None
        self.signed_url_calls: list[dict] = []
        self.metadata: dict = {}

    def upload_from_string(self, data, content_type, timeout=None):
        self.uploaded_bytes = data
        self.uploaded_content_type = content_type
        self.bucket.blobs[self.name] = self

    def generate_signed_url(self, **kwargs):
        self.signed_url_calls.append(kwargs)
        return f"https://signed.example/{self.bucket.name}/{self.name}?exp=test"


class _FakeBucket:
    def __init__(self, name: str):
        self.name = name
        self.blobs: dict[str, _FakeBlob] = {}
        self._exists = True

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self)

    def exists(self) -> bool:
        return self._exists


class _FakeStorage:
    def __init__(self):
        self.bucket_obj = _FakeBucket("my-bucket")

    def bucket(self, name: str) -> _FakeBucket:
        assert name == "my-bucket"
        return self.bucket_obj


@pytest.mark.asyncio
async def test_upload_and_sign_returns_url_and_path():
    from api.storage import StorageBackend

    fake = _FakeStorage()
    backend = StorageBackend(
        client=fake,
        bucket_name="my-bucket",
        object_prefix="tts",
        signed_url_expiry_minutes=30,
    )
    audio = b"RIFF....fake-wav"
    result = await backend.upload_wav(audio_bytes=audio, content_type="audio/wav")

    # path shape: tts/YYYY-MM-DD/<uuid>.wav
    parts = result.gcs_object.split("/")
    assert parts[0] == "tts"
    assert len(parts[1]) == 10 and parts[1].count("-") == 2  # YYYY-MM-DD
    assert parts[2].endswith(".wav")

    blob = fake.bucket_obj.blobs[result.gcs_object]
    assert blob.uploaded_bytes == audio
    assert blob.uploaded_content_type == "audio/wav"
    assert result.audio_url.startswith("https://signed.example/")
    assert result.audio_url_expires_at > dt.datetime.now(dt.timezone.utc)
    # 30-minute expiry
    delta = result.audio_url_expires_at - dt.datetime.now(dt.timezone.utc)
    assert 25 * 60 < delta.total_seconds() < 32 * 60


@pytest.mark.asyncio
async def test_signed_url_uses_v4_get_with_correct_expiry():
    from api.storage import StorageBackend

    fake = _FakeStorage()
    backend = StorageBackend(
        client=fake,
        bucket_name="my-bucket",
        object_prefix="tts",
        signed_url_expiry_minutes=45,
    )
    audio = b"RIFF....fake-wav"
    result = await backend.upload_wav(audio_bytes=audio, content_type="audio/wav")
    blob = fake.bucket_obj.blobs[result.gcs_object]
    assert blob.signed_url_calls
    call = blob.signed_url_calls[0]
    assert call["version"] == "v4"
    assert call["method"] == "GET"
    assert int(call["expiration"].total_seconds()) == 45 * 60


@pytest.mark.asyncio
async def test_upload_failure_raises_storage_unavailable():
    from api.errors import StorageUnavailableError
    from api.storage import StorageBackend

    class _BoomBucket(_FakeBucket):
        def blob(self, name):
            blob = super().blob(name)
            def boom(*a, **kw):
                raise RuntimeError("gcs down")
            blob.upload_from_string = boom  # type: ignore[assignment]
            return blob

    class _BoomStorage(_FakeStorage):
        def __init__(self):
            self.bucket_obj = _BoomBucket("my-bucket")

    backend = StorageBackend(
        client=_BoomStorage(),
        bucket_name="my-bucket",
        object_prefix="tts",
        signed_url_expiry_minutes=30,
    )
    with pytest.raises(StorageUnavailableError):
        await backend.upload_wav(audio_bytes=b"x", content_type="audio/wav")


@pytest.mark.asyncio
async def test_check_bucket_returns_true_when_exists():
    from api.storage import StorageBackend

    fake = _FakeStorage()
    backend = StorageBackend(
        client=fake, bucket_name="my-bucket",
        object_prefix="tts", signed_url_expiry_minutes=30,
    )
    assert await backend.check_bucket() is True

    fake.bucket_obj._exists = False
    assert await backend.check_bucket() is False
