"""FastAPI app entry point.

Owns the lifecycle of the httpx AsyncClient, the GCS storage Client, and
the SpeakersCache. Routes pull these from app.state via dependencies.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from google.cloud import storage as gcs_storage

# Load .env into os.environ at import time so google.auth.default() can see
# GOOGLE_APPLICATION_CREDENTIALS for local SA-key signing. pydantic-settings
# reads .env into the Settings object only — it does NOT export to os.environ,
# so the GCS auth layer (which calls os.getenv directly) would otherwise fall
# through to gcloud user ADC and break signed URLs. On Cloud Run / GKE the SA
# path env var is unset and Workload Identity takes over via the metadata
# server instead. Missing .env is a no-op.
load_dotenv()

from api.config import get_settings
from api.errors import register_exception_handlers, request_id_middleware
from api.logging_setup import setup_logging
from api.modal_client import ModalClient
from api.routes import router
from api.speakers import SpeakersCache
from api.storage import StorageBackend

logger = logging.getLogger("orpheus_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    timeout = httpx.Timeout(
        connect=settings.modal_connect_timeout_seconds,
        read=settings.modal_request_timeout_seconds,
        write=10.0,
        pool=10.0,
    )
    transport = httpx.AsyncHTTPTransport(retries=1)
    http = httpx.AsyncClient(
        base_url=settings.orpheus_modal_url,
        timeout=timeout,
        transport=transport,
    )
    modal = ModalClient(client=http, retry_backoff_seconds=settings.modal_retry_backoff_seconds)
    speakers = SpeakersCache(modal=modal, ttl_seconds=settings.speakers_cache_ttl_seconds)
    gcs_client = gcs_storage.Client()
    storage = StorageBackend(
        client=gcs_client,
        bucket_name=settings.gcs_bucket_name,
        object_prefix=settings.gcs_object_prefix,
        signed_url_expiry_minutes=settings.gcs_signed_url_expiry_minutes,
    )

    app.state.settings = settings
    app.state.http = http
    app.state.modal = modal
    app.state.speakers = speakers
    app.state.storage = storage

    await speakers.try_warm()
    logger.info(
        "startup_complete",
        extra={
            "modal_url": settings.orpheus_modal_url,
            "bucket": settings.gcs_bucket_name,
            "speakers_warm": speakers.is_warm,
        },
    )
    try:
        yield
    finally:
        await http.aclose()
        # google-cloud-storage holds a requests.Session + connection pool.
        # close() is idempotent and safe even if it was never opened.
        try:
            gcs_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gcs_client_close_failed: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Orpheus-3B TTS API",
        version="0.1.0",
        description=(
            "Multilingual, multi-speaker TTS gateway. Calls a Modal-deployed "
            "vLLM inference server, uploads audio to GCS, and returns a "
            "presigned URL that expires in 30 minutes."
        ),
        lifespan=lifespan,
    )
    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
