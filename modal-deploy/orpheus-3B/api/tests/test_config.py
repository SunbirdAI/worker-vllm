import os
from pathlib import Path

import pytest
from pydantic import ValidationError


def _clear_env(monkeypatch, keep=()):
    """Remove every ORPHEUS_/GCS_/GOOGLE_APPLICATION_CREDENTIALS env var,
    except those in `keep`, before instantiating Settings."""
    for var in list(os.environ):
        if var.startswith(("ORPHEUS_", "GCS_", "MODAL_", "API_", "SPEAKERS_")):
            if var not in keep:
                monkeypatch.delenv(var, raising=False)
        if var == "GOOGLE_APPLICATION_CREDENTIALS" and var not in keep:
            monkeypatch.delenv(var, raising=False)


def test_settings_load_with_required_vars(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    creds = tmp_path / "sa.json"
    creds.write_text("{}")
    monkeypatch.setenv("ORPHEUS_MODAL_URL", "https://example.modal.run")
    monkeypatch.setenv("GCS_BUCKET_NAME", "my-bucket")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))

    from api.config import Settings
    s = Settings()
    assert s.orpheus_modal_url == "https://example.modal.run"
    assert s.gcs_bucket_name == "my-bucket"
    assert s.google_application_credentials == str(creds)
    assert s.gcs_signed_url_expiry_minutes == 30
    assert s.max_batch_size == 16


def test_settings_missing_required_raises(monkeypatch):
    _clear_env(monkeypatch)
    from api.config import Settings
    # _env_file=None disables .env reading for this test so it stays hermetic
    # even when a developer has a real .env in the cwd.
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    # error should name at least one of the missing required fields
    msg = str(excinfo.value).lower()
    assert "orpheus_modal_url" in msg or "gcs_bucket_name" in msg


def test_settings_credentials_file_must_exist(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ORPHEUS_MODAL_URL", "https://example.modal.run")
    monkeypatch.setenv("GCS_BUCKET_NAME", "my-bucket")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "does-not-exist.json")
    )
    from api.config import Settings
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert "google_application_credentials" in str(excinfo.value).lower()


def test_settings_overrides(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    creds = tmp_path / "sa.json"
    creds.write_text("{}")
    monkeypatch.setenv("ORPHEUS_MODAL_URL", "https://example.modal.run")
    monkeypatch.setenv("GCS_BUCKET_NAME", "my-bucket")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))
    monkeypatch.setenv("GCS_SIGNED_URL_EXPIRY_MINUTES", "60")
    monkeypatch.setenv("MAX_BATCH_SIZE", "32")

    from api.config import Settings
    s = Settings()
    assert s.gcs_signed_url_expiry_minutes == 60
    assert s.max_batch_size == 32
