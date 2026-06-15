# Orpheus-3B TTS FastAPI Service

A thin HTTP gateway in front of the Modal-deployed Orpheus-3B inference app.

- Calls Modal `/tts`, `/tts/batch`, `/speakers`, `/health` via async httpx.
- Uploads each generated WAV to Google Cloud Storage.
- Returns a v4 presigned URL (default 30-minute expiry) plus metadata and latency timings.
- Validates `speaker_id` (and optional `language`) against the live catalog up front, so bad inputs fail at the API tier with a 400 instead of consuming GPU time on Modal.

## Quick start

```bash
cd orpheus-3B/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Modal URL, GCS bucket name, and GCP service account JSON path
cd ..   # back to orpheus-3B/
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI: <http://localhost:8000/docs>.

## Environment variables

See `.env.example`. Required:

- `ORPHEUS_MODAL_URL` — base URL of the deployed Modal app.
- `GCS_BUCKET_NAME` — existing GCS bucket; the API writes objects, it does **not** create/delete buckets or set lifecycle rules.
- `GOOGLE_APPLICATION_CREDENTIALS` — absolute path to a service account JSON key file. The SA needs `roles/storage.objectCreator` on the bucket (for uploads) and `roles/storage.objectViewer` (or `storage.buckets.get`) for `/healthz`'s readiness probe.

Optional vars (with defaults) are documented inline in `.env.example`.

## Endpoints

### `POST /tts` — single synthesis

```bash
curl -s -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Mwattu, oli otya?", "speaker_id": "salt_lug_0001"}' | jq
```

Response:
```json
{
  "audio_url": "https://storage.googleapis.com/...signed...",
  "audio_url_expires_at": "2026-05-12T15:30:00Z",
  "speaker_id": "salt_lug_0001",
  "language": "lug",
  "sample_rate": 24000,
  "duration_seconds": 2.45,
  "audio_size_bytes": 117648,
  "gcs_object": "tts/2026-05-12/<uuid>.wav",
  "request_id": "<uuid>",
  "timings_ms": {
    "inference_ms": 1820.5,
    "upload_ms": 234.1,
    "signed_url_ms": 12.0,
    "total_ms": 2095.6
  }
}
```

### `POST /tts/batch` — multi-item synthesis

```bash
curl -s -X POST http://localhost:8000/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      {"text": "Hi there",      "speaker_id": "salt_eng_0001", "language": "eng"},
      {"text": "Mwattu, oli otya?", "speaker_id": "salt_lug_0001", "language": "lug"},
      {"text": "Habari yako?",  "speaker_id": "waxal_swa_0006", "language": "swa"}
    ]
  }' | jq
```

Per-item results carry `status: "ok"` or `status: "error"`. The batch returns 200 if ≥1 item succeeded, 502 if all failed.

### `GET /speakers` and `GET /speakers/{language}`

```bash
curl -s http://localhost:8000/speakers | jq
curl -s http://localhost:8000/speakers/lug | jq
```

### `GET /health` and `/healthz`

- `/health` — liveness; cheap, no upstream calls. Always 200 when the process is up.
- `/healthz` — readiness; pings Modal `/health` and probes GCS. 200 when both healthy, 503 with per-upstream status otherwise.

## Selecting a speaker (recommended client flow)

1. `GET /speakers` — get the full `by_language` map.
2. Show a language picker; once the user picks (say) `lug`, call `GET /speakers/lug`.
3. Show a speaker picker scoped to that language.
4. Submit `POST /tts` with `language` set, for an extra safety check.

## Python client example (httpx)

```python
import httpx

with httpx.Client(base_url="http://localhost:8000", timeout=httpx.Timeout(180.0, connect=10.0)) as c:
    r = c.post("/tts", json={
        "text": "Mwattu, oli otya?",
        "speaker_id": "salt_lug_0001",
        "language": "lug",
    })
    r.raise_for_status()
    data = r.json()
    print("download:", data["audio_url"])
    print(f"duration: {data['duration_seconds']:.2f}s")
    print(f"inference: {data['timings_ms']['inference_ms']:.0f} ms")
```

## Running tests

### Unit / integration suite (hermetic)

```bash
cd orpheus-3B
pytest api/tests/ -v
```

73 tests, all hermetic — no real Modal, no real GCS (httpx `MockTransport` + an in-memory fake storage client). This is what runs in CI.

### Manual scripts against a live deployment

Three small clients live in `orpheus-3B/api/scripts/`. They all read `API_BASE` from the environment (default `http://localhost:8000`) and exercise the real running API + real Modal + real GCS. Useful for sanity-checking a fresh deploy or new speaker before relying on it.

Activate the venv first:

```bash
cd /path/to/Qwen3-TTS
source orpheus-3B/api/.venv/bin/activate
```

#### `smoke.py` — minimal end-to-end check

Hits `/health`, `/healthz`, `/speakers`, then a single `/tts` (Luganda) so you can confirm in under a minute that all four paths work.

```bash
python orpheus-3B/api/scripts/smoke.py

# different host
API_BASE=https://my-api.example.com python orpheus-3B/api/scripts/smoke.py
```

#### `test_english_tts.py` — single + batch with an English speaker

Hits `/tts` with one sentence and `/tts/batch` with five sentences using `salt_eng_0002`. Prints `request_id`, durations, audio sizes, presigned URLs, expiry times, and stage-by-stage timings (`inference_ms` / `upload_ms` / `signed_url_ms` / `total_ms`).

```bash
python orpheus-3B/api/scripts/test_english_tts.py

# also stream each WAV to /tmp/orpheus-english-tts/ so you can listen to them
DOWNLOAD=1 python orpheus-3B/api/scripts/test_english_tts.py
```

Useful env flags:

| Var | Default | Purpose |
|---|---|---|
| `API_BASE` | `http://localhost:8000` | Different host |
| `DOWNLOAD` | (unset) | Set to `1` to save each WAV locally |
| `OUT_DIR` | `/tmp/orpheus-english-tts` | Where to save WAVs when `DOWNLOAD=1` |

#### `test_long_text.py` — long-form text via split + batch + stitch

Recommended pattern for paragraphs / articles / scripts: each `/tts` call is bounded by `max_tokens` (default 1200, hard cap 4096), so a long single input can produce truncated audio. This script:

1. Splits the input on sentence boundaries (`. ! ?`); overly long sentences are further split on `,` / `;` / spaces so no chunk exceeds 500 chars.
2. Groups sentences into batches of `BATCH_SIZE` (default 8, must stay ≤ server `MAX_BATCH_SIZE`) and sends each batch to `/tts/batch` — vLLM continuous batching means N sentences in one batch is roughly 1.5× the latency of one sentence, not N×.
3. Downloads each per-item WAV (presigned URL), asserts 24 kHz / mono / 16-bit, and concatenates the PCM frames with a configurable silence between sentences (default 250 ms).
4. Writes one stitched WAV using stdlib `wave` — no extra dependencies.

```bash
# inline text
python orpheus-3B/api/scripts/test_long_text.py \
  --speaker salt_eng_0002 --language eng \
  --text "Good morning everyone. Today we will cover three things. First, last week's progress. Second, the upcoming launch and timeline. Finally, open questions about marketing."

# from a file
python orpheus-3B/api/scripts/test_long_text.py \
  --speaker salt_eng_0002 --language eng \
  --file my_article.txt \
  --out /tmp/article.wav
```

Useful env flags:

| Var | Default | Purpose |
|---|---|---|
| `API_BASE` | `http://localhost:8000` | Different host |
| `BATCH_SIZE` | `8` | Items per `/tts/batch` call. Keep ≤ server `MAX_BATCH_SIZE`. |
| `SILENCE_MS` | `250` | Silence inserted between sentences. Bump to 400 for slower pacing. |

One failed sentence is logged and skipped — the rest still produce audio. Exit code is `1` if any sentence failed.

## Errors

All error responses share this shape:

```json
{"error": "<machine-code>", "detail": "<human>", "request_id": "<uuid>"}
```

| Status | Code | When |
|---|---|---|
| 400 | `invalid_speaker` | `speaker_id` not in catalog |
| 400 | `unknown_language` | `language` not in catalog |
| 400 | `invalid_speaker_for_language` | `speaker_id` exists but is from a different language than `language` |
| 422 | `invalid_request` | Pydantic validation (e.g. text too long, batch empty) |
| 502 | `upstream_unavailable` | Modal failed twice |
| 502 | `storage_unavailable` | GCS upload or signing failed |
| 504 | `upstream_timeout` | Modal `read` timeout twice (usually a long cold-start) |
| 503 | (no code; `/healthz` only) | Readiness probe reporting degraded |

`X-Request-ID` is set on every response. Pass `X-Request-ID: <yours>` to thread your own ID through logs.

## Out of scope

- API-level auth (deploy behind a gateway).
- Streaming response (`/tts/stream` is planned separately).
- Cloud deploy (Dockerfile, Cloud Run/GKE manifests) — separate spec.
