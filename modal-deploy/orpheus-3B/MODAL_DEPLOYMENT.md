# Deploying Orpheus-3B Multilingual TTS to Modal

This guide walks through deploying the finetuned multilingual Orpheus-3B TTS
([`sunbird/orpheus-3b-tts-multilingual`](https://huggingface.co/sunbird/orpheus-3b-tts-multilingual))
to [Modal](https://modal.com) as a vLLM-backed, FastAPI HTTP service.

The deployment exposes:

| Endpoint            | Method | Purpose                                                                 |
|---------------------|--------|-------------------------------------------------------------------------|
| `/tts`              | POST   | Single utterance, returns `audio/wav`                                   |
| `/tts/batch`        | POST   | Multi-utterance batched synthesis (multi-speaker), returns base64 wavs  |
| `/speakers`         | GET    | Static catalog of speaker IDs grouped by language                       |
| `/health`           | GET    | Liveness probe with model + sample-rate info                            |

All endpoints accept a `speaker_id` per request, so a single replica serves
any speaker the finetuned checkpoint knows (multi-language, multi-speaker).

Files used by this guide:
- [orpheus-3B/modal_deploy.py](modal_deploy.py) — the Modal app
- [orpheus-3B/Orpheus_3B_Sunbird_Luganda_vLLM_Inference.ipynb](Orpheus_3B_Sunbird_Luganda_vLLM_Inference.ipynb) — reference notebook the script mirrors

---

## 1. Prerequisites

### 1.1 Tooling

```bash
# Modal CLI (also used to manage secrets, volumes, deploys)
pip install -U modal

# Log in once per workstation (opens a browser tab)
modal token new
```

You will also need a HuggingFace account with read access to the Orpheus
checkpoint (the published Sunbird repos are public, so a generic read token
works).

### 1.2 Modal secret for HuggingFace

The script expects a Modal secret named `huggingface-secret` containing an
`HF_TOKEN`. Create it once:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx_your_read_token
```

Confirm with:

```bash
modal secret list | grep huggingface-secret
```

### 1.3 (Optional) Override defaults at deploy time

Defaults baked into the script can be overridden with environment variables
that Modal reads from your shell at deploy time:

| Env var                       | Default                                  | Effect                                |
|-------------------------------|------------------------------------------|---------------------------------------|
| `ORPHEUS_MODEL_ID`            | `sunbird/orpheus-3b-tts-multilingual`    | HF repo or local path to load         |
| `ORPHEUS_DEFAULT_SPEAKER`     | `salt_lug_0001`                          | Used when a request omits `speaker_id`|
| `ORPHEUS_MAX_MODEL_LEN`       | `4096`                                   | vLLM KV-cache budget                  |
| `ORPHEUS_GPU`                 | `L40S`                                   | Modal GPU type — `A10G`, `L40S`, `H100`|
| `ORPHEUS_MAX_INPUTS`          | `16`                                     | Concurrent requests per replica       |

Example:

```bash
ORPHEUS_GPU=H100 ORPHEUS_MAX_INPUTS=32 modal deploy orpheus-3B/modal_deploy.py
```

---

## 2. (Recommended) Warm the HuggingFace cache volume

The first time the container starts it will download ~6 GB of model weights.
A one-shot warmup writes them into a persistent Modal Volume so future cold
starts skip the download:

```bash
modal run orpheus-3B/modal_deploy.py::download_model
```

This creates two volumes if missing: `orpheus-hf-cache` and `orpheus-vllm-cache`.
The first stores HuggingFace snapshots; the second caches vLLM's compiled
CUDA graphs across container restarts.

To warm a custom checkpoint, pass `--model-id`:

```bash
modal run orpheus-3B/modal_deploy.py::download_model --model-id patrickcmd/orpheus-3b-tts-multilingual
```

---

## 3. Deploy

```bash
modal deploy orpheus-3B/modal_deploy.py
```

Modal prints the public URL when the deploy completes. It looks like:

```
https://<workspace>--orpheus-3b-tts-orpheustts-web.modal.run
```

This is the **base URL** for the four endpoints below.

The first request after deploy triggers a cold start: pulling the image and
loading the model (~30–60 s on a warm cache, several minutes on a cold one).
Subsequent requests are near-instant until the scaledown window (15 minutes
of idle) elapses.

---

## 4. Test the deployed endpoints

Replace `BASE_URL` below with the URL printed by `modal deploy`.

### 4.1 Health check

```bash
BASE_URL='https://<workspace>--orpheus-3b-tts-orpheustts-web.modal.run'

curl -s "$BASE_URL/health" | jq
```

Expected output:

```json
{
  "status": "ok",
  "model": "sunbird/orpheus-3b-tts-multilingual",
  "max_model_len": 4096,
  "sample_rate": 24000
}
```

### 4.2 Speaker catalog

```bash
curl -s "$BASE_URL/speakers" | jq
```

The catalog is static (edit `SPEAKERS_BY_LANGUAGE` in `modal_deploy.py` to
reflect the speakers you actually finetuned on). Any speaker tag the
checkpoint was trained on will work even if it is not listed here.

### 4.3 Single utterance (`audio/wav` response)

```bash
curl -s -X POST "$BASE_URL/tts" \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Mwattu, oli otya?",
    "speaker_id": "salt_lug_0001",
    "seed": 42
  }' \
  --output luganda_hello.wav

afplay luganda_hello.wav   # macOS — or `aplay` on Linux
```

Response headers carry the duration and speaker:

```bash
curl -sI -X POST "$BASE_URL/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Mwattu, oli otya?","speaker_id":"salt_lug_0001"}'
```

```
HTTP/2 200
content-type: audio/wav
x-sample-rate: 24000
x-duration-seconds: 1.583
x-speaker-id: salt_lug_0001
```

### 4.4 Batched multi-speaker synthesis

`POST /tts/batch` accepts an `items` array, each carrying its own
`speaker_id`. vLLM batches them all through one GPU pass — much faster
than calling `/tts` N times.

```bash
curl -s -X POST "$BASE_URL/tts/batch" \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      {"text": "Mwattu, oli otya?",       "speaker_id": "salt_lug_0001"},
      {"text": "Habari ya asubuhi.",      "speaker_id": "waxal_swa_0006"},
      {"text": "Itye ma ber?",            "speaker_id": "salt_ach_0001"},
      {"text": "Abu ngesi ilip itunga.",  "speaker_id": "salt_teo_0001"}
    ]
  }' \
  | jq -r '.results[] | "\(.speaker_id) \(.duration_sec)s -> \(.audio_wav_b64[0:40])..."'
```

Response shape:

```json
{
  "results": [
    {
      "text": "Mwattu, oli otya?",
      "speaker_id": "salt_lug_0001",
      "sample_rate": 24000,
      "duration_sec": 1.58,
      "audio_wav_b64": "UklGRiQAAABXQVZFZm10IBAAAAAB..."
    }
  ]
}
```

### 4.5 Python client (single + batch)

```python
"""orpheus_modal_client.py — minimal Python client for the deployed service."""
import base64
import requests

BASE_URL = "https://<workspace>--orpheus-3b-tts-orpheustts-web.modal.run"


def tts(text: str, speaker_id: str, out_path: str, **decode) -> None:
    r = requests.post(
        f"{BASE_URL}/tts",
        json={"text": text, "speaker_id": speaker_id, **decode},
        timeout=120,
    )
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"{speaker_id}: {text!r} -> {out_path}"
          f" ({r.headers.get('x-duration-seconds')}s)")


def tts_batch(items: list[dict], out_dir: str = ".") -> None:
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    r = requests.post(f"{BASE_URL}/tts/batch", json={"items": items}, timeout=300)
    r.raise_for_status()
    for i, result in enumerate(r.json()["results"]):
        wav_bytes = base64.b64decode(result["audio_wav_b64"])
        path = f"{out_dir}/{i:02d}_{result['speaker_id']}.wav"
        with open(path, "wb") as f:
            f.write(wav_bytes)
        print(f"{path}: {result['duration_sec']:.2f}s")


if __name__ == "__main__":
    tts("Mwattu, oli otya?", "salt_lug_0001", "hello_lug.wav", seed=42)
    tts_batch(
        [
            {"text": "Mwattu, oli otya?",  "speaker_id": "salt_lug_0001"},
            {"text": "Habari ya asubuhi.", "speaker_id": "waxal_swa_0006"},
            {"text": "Itye ma ber?",       "speaker_id": "salt_ach_0001"},
        ],
        out_dir="modal_out",
    )
```

Run:

```bash
python orpheus_modal_client.py
```

---

## 5. Decoding parameters

Same knobs as the inference notebook — all optional on the request:

| Field                | Default | Notes                                                                              |
|----------------------|---------|------------------------------------------------------------------------------------|
| `temperature`        | `0.6`   | Lower = more deterministic prosody; higher = more variation, can be unstable      |
| `top_p`              | `0.95`  | Nucleus sampling cutoff                                                            |
| `repetition_penalty` | `1.1`   | Discourages duplicate audio tokens (long pauses / chant artifacts)                |
| `max_tokens`         | `1200`  | Hard cap on audio tokens generated (~1200 = ~10 s of audio)                       |
| `seed`               | `null`  | Pass an int for reproducible output                                                |

Run a sweep with the notebook's `#4` cell pattern if you want to tune for a
particular voice/text combo before locking these values into the client.

---

## 6. Scaling, GPU choice, and cost

| GPU    | VRAM  | Cost relative | When to use                                              |
|--------|-------|---------------|----------------------------------------------------------|
| `A10G` | 24 GB | low           | Light traffic, latency-tolerant                          |
| `L40S` | 48 GB | medium        | **Default.** Best cost/throughput for Orpheus-3B bf16    |
| `H100` | 80 GB | high          | Highest throughput, large batches, low p95 latency       |

Other knobs in the script:

- `@modal.concurrent(max_inputs=16)` — how many in-flight requests a single
  replica accepts before Modal starts a second one. Raising this leans on
  vLLM's continuous batching for throughput; lower it if you see GPU OOM.
- `scaledown_window=15 * MINUTES` — how long a replica stays warm with no
  traffic. Shorten to save cost; lengthen to avoid cold starts during
  bursty workloads.
- `min_containers=0` — set to `1` to keep one replica always warm (no cold
  starts, but you pay 24/7).
- `gpu_memory_utilization=0.85` in `LLM(...)` — drop to `0.70` if you hit
  OOM during model load on smaller GPUs.

---

## 7. Operations

### 7.1 Tail logs

```bash
modal app logs orpheus-3b-tts
```

### 7.2 Stop the app (no further requests served)

```bash
modal app stop orpheus-3b-tts
```

### 7.3 Inspect / list / delete the cache volumes

```bash
modal volume list
modal volume ls orpheus-hf-cache
modal volume rm  orpheus-hf-cache   # wipes the HF cache; next deploy redownloads
```

### 7.4 Redeploy after editing `modal_deploy.py`

```bash
modal deploy orpheus-3B/modal_deploy.py
```

Modal does a rolling update — the old replicas keep serving until the new
ones are healthy.

---

## 8. Troubleshooting

### `AssertionError: ... org_vocab_size` on container start

Your checkpoint's `config.json` has `vocab_size = 156940` but the actual
embedding weight is `[156939, 3072]`. This is the Unsloth `push_to_hub_merged`
off-by-one. The script already passes `hf_overrides={"vocab_size": 156939}`
to vLLM to fix this. If you're loading a third-party Orpheus checkpoint and
still hit the error, edit `VOCAB_SIZE` at the top of `modal_deploy.py` to
match the actual embedding shape.

(The training notebooks for both single-speaker and multilingual finetunes
now patch `config.json` to `156939` both locally and on the hub after
`push_to_hub_merged`, so newly trained checkpoints don't need the override.)

### Long cold starts on first request

The very first deploy pulls the container image and downloads the model.
Run `modal run orpheus-3B/modal_deploy.py::download_model` once before
deploying to push the weights into the persistent volume.

### Repeated 502s under load

Bump `@modal.concurrent(max_inputs=...)` carefully — vLLM's per-request KV
budget is fixed, so very high concurrency on a single GPU eventually
overflows. Either lower `MAX_MODEL_LEN` or `max_tokens`, or let Modal scale
horizontally by leaving `max_inputs` modest (8–16) and serving more replicas.

### `HF_TOKEN not set` / 401 from HuggingFace

The `huggingface-secret` Modal secret is missing or the token expired.
Re-run `modal secret create huggingface-secret HF_TOKEN=hf_...`.

---

## 9. What this deployment does *not* do

- No streaming. The endpoint waits for the full utterance before returning.
  If you need token-streaming TTS, return Server-Sent Events from a
  generator inside the FastAPI route and decode SNAC in fixed-size windows.
- No request authentication. Add a `@web_app.middleware("http")` check on a
  shared API key, or wrap the URL behind your own auth proxy.
- No long-form chunking. Inputs longer than ~30–40 seconds of speech may
  hit `max_tokens`. Chunk on sentence boundaries on the client side for
  long passages.
