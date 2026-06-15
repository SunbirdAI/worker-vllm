"""HTTP route handlers."""

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from api.errors import (
    InvalidSpeakerError,
    InvalidSpeakerForLanguageError,
    StorageUnavailableError,
    UnknownLanguageError,
    get_request_id,
)
from api.models import (
    BatchTimings,
    HealthResponse,
    LanguageSpeakersResponse,
    ReadinessResponse,
    SpeakersResponse,
    Timings,
    TTSBatchItemResult,
    TTSBatchRequest,
    TTSBatchResponse,
    TTSRequest,
    TTSResponse,
)

router = APIRouter()


# ----- /speakers -----

@router.get(
    "/speakers",
    response_model=SpeakersResponse,
    summary="List all speakers grouped by language.",
    description=(
        "Returns the full speaker catalog. Use this to populate a "
        "client-side picker. `total` and `languages` are derived "
        "convenience fields."
    ),
)
async def get_speakers(request: Request) -> SpeakersResponse:
    cat = await request.app.state.speakers.get()
    return SpeakersResponse(default=cat.default, by_language=cat.by_language)


@router.get(
    "/speakers/{language}",
    response_model=LanguageSpeakersResponse,
    summary="List speakers for one language.",
    description=(
        "Convenience endpoint for two-step pickers: pick a language, then "
        "list its speakers. Returns 404 unknown_language if the code is "
        "not in the catalog."
    ),
    responses={
        404: {"description": "language code not found in catalog"},
    },
)
async def get_speakers_for_language(
    language: str, request: Request
) -> LanguageSpeakersResponse:
    cat = await request.app.state.speakers.get()
    if language not in cat.by_language:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_language",
                "detail": (
                    f"language '{language}' not supported; "
                    f"supported: {sorted(cat.by_language)}"
                ),
            },
        )
    return LanguageSpeakersResponse(
        language=language, speakers=cat.by_language[language]
    )


# ----- /tts -----

@router.post(
    "/tts",
    response_model=TTSResponse,
    summary="Synthesize speech for one input.",
    description=(
        "Calls the Modal vLLM inference app, uploads the generated WAV to "
        "Google Cloud Storage, and returns a v4 presigned download URL "
        "valid for the configured expiry window (default 30 minutes), "
        "together with metadata and stage-by-stage latency timings."
    ),
    responses={
        400: {"description": "invalid_speaker | unknown_language | invalid_speaker_for_language"},
        422: {"description": "request validation error"},
        502: {"description": "upstream_unavailable | storage_unavailable"},
        504: {"description": "upstream_timeout"},
    },
)
async def tts(req: Annotated[TTSRequest, Body()], request: Request) -> TTSResponse:
    speakers = request.app.state.speakers
    modal = request.app.state.modal
    storage = request.app.state.storage

    await speakers.validate_speaker(req.speaker_id, language=req.language)

    t_total = time.monotonic()
    t_inf = time.monotonic()
    audio = await modal.tts(
        text=req.text,
        speaker_id=req.speaker_id,
        seed=req.seed,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_tokens=req.max_tokens,
    )
    inference_ms = (time.monotonic() - t_inf) * 1000.0

    upload = await storage.upload_wav(
        audio_bytes=audio.audio_bytes, content_type="audio/wav"
    )

    total_ms = (time.monotonic() - t_total) * 1000.0
    resolved_language = await speakers.language_for(req.speaker_id)

    return TTSResponse(
        audio_url=upload.audio_url,
        audio_url_expires_at=upload.audio_url_expires_at,
        speaker_id=req.speaker_id,
        language=resolved_language,
        sample_rate=audio.sample_rate,
        duration_seconds=audio.duration_seconds,
        chunks=audio.chunks,
        audio_size_bytes=upload.audio_size_bytes,
        gcs_object=upload.gcs_object,
        request_id=get_request_id(),
        timings_ms=Timings(
            inference_ms=inference_ms,
            upload_ms=upload.upload_ms,
            signed_url_ms=upload.signed_url_ms,
            total_ms=total_ms,
        ),
    )


@router.post(
    "/tts/batch",
    response_model=TTSBatchResponse,
    summary="Synthesize speech for a batch of inputs.",
    description=(
        "Calls Modal's batched inference endpoint (a single vLLM "
        "continuous-batched pass) and uploads each generated WAV to GCS "
        "in parallel. Per-item failures are reported in the response with "
        "`status: \"error\"`; the request as a whole returns 200 if at "
        "least one item succeeds, 502 if every item failed."
    ),
    responses={
        400: {"description": "invalid_speaker | unknown_language | invalid_speaker_for_language (any item)"},
        422: {"description": "request validation error"},
        502: {"description": "all items failed upstream or storage"},
        504: {"description": "upstream_timeout"},
    },
)
async def tts_batch(
    req: Annotated[TTSBatchRequest, Body()], request: Request
) -> TTSBatchResponse:
    settings = request.app.state.settings
    speakers = request.app.state.speakers
    modal = request.app.state.modal
    storage = request.app.state.storage

    if len(req.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request",
                "detail": (
                    f"batch size {len(req.items)} exceeds MAX_BATCH_SIZE "
                    f"{settings.max_batch_size}"
                ),
            },
        )

    # Pre-flight validation — fail fast for the entire batch on first bad item.
    for idx, item in enumerate(req.items):
        try:
            await speakers.validate_speaker(item.speaker_id, language=item.language)
        except (InvalidSpeakerError, UnknownLanguageError, InvalidSpeakerForLanguageError) as exc:
            exc.detail = f"item index {idx}: {exc.detail}"
            raise

    t_total = time.monotonic()
    t_inf = time.monotonic()
    audios = await modal.tts_batch(
        [
            {
                "text": it.text,
                "speaker_id": it.speaker_id,
                "seed": it.seed,
                "temperature": it.temperature,
                "top_p": it.top_p,
                "repetition_penalty": it.repetition_penalty,
                "max_tokens": it.max_tokens,
            }
            for it in req.items
        ]
    )
    inference_ms = (time.monotonic() - t_inf) * 1000.0

    t_up = time.monotonic()
    uploads = await asyncio.gather(
        *[
            storage.upload_wav(audio_bytes=a.audio_bytes, content_type="audio/wav")
            for a in audios
        ],
        return_exceptions=True,
    )
    upload_ms = (time.monotonic() - t_up) * 1000.0

    results: list[TTSBatchItemResult] = []
    ok_count = 0
    for i, (item, audio, up) in enumerate(zip(req.items, audios, uploads)):
        if isinstance(up, Exception):
            code = getattr(up, "error_code", "storage_unavailable")
            results.append(
                TTSBatchItemResult(
                    index=i,
                    status="error",
                    speaker_id=item.speaker_id,
                    error_code=code,
                    error_detail=str(up),
                )
            )
            continue
        ok_count += 1
        results.append(
            TTSBatchItemResult(
                index=i,
                status="ok",
                speaker_id=item.speaker_id,
                language=await speakers.language_for(item.speaker_id),
                audio_url=up.audio_url,
                audio_url_expires_at=up.audio_url_expires_at,
                sample_rate=audio.sample_rate,
                duration_seconds=audio.duration_seconds,
                audio_size_bytes=up.audio_size_bytes,
                gcs_object=up.gcs_object,
                request_id=get_request_id(),
            )
        )

    total_ms = (time.monotonic() - t_total) * 1000.0
    if ok_count == 0:
        raise StorageUnavailableError(
            f"all {len(req.items)} batch items failed during upload"
        )

    return TTSBatchResponse(
        results=results,
        timings_ms=BatchTimings(
            inference_ms=inference_ms, upload_ms=upload_ms, total_ms=total_ms
        ),
        request_id=get_request_id(),
    )


# ----- /health / /healthz -----

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe (no upstream calls).",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/healthz",
    response_model=ReadinessResponse,
    summary="Readiness probe — checks Modal and GCS.",
    responses={503: {"description": "one or more upstreams unreachable"}},
)
async def healthz(request: Request):
    modal_ok = await request.app.state.modal.health()
    gcs_ok = await request.app.state.storage.check_bucket()
    upstreams = {
        "modal": "ok" if modal_ok else "unreachable",
        "gcs": "ok" if gcs_ok else "unreachable",
    }
    if modal_ok and gcs_ok:
        return ReadinessResponse(status="ok", upstreams=upstreams)
    return JSONResponse(
        status_code=503,
        content=ReadinessResponse(
            status="degraded", upstreams=upstreams
        ).model_dump(),
    )
