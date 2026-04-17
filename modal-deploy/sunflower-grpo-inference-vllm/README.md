# Sunflower 14B GRPO — Modal + vLLM Deployment

Serves the **`jq/sunflower-14b-bs64-lr1e-4-250919`** base model with the
**`jq/sunflower-14b-grpo-combined`** LoRA adapter on [Modal](https://modal.com)
using **vLLM**, exposing both blocking and streaming HTTP endpoints.

Files:

- [sunflower_grpo_vllm_modal.py](sunflower_grpo_vllm_modal.py) — Modal app (image, volume, vLLM class, FastAPI endpoints).
- [fastapi_app.py](fastapi_app.py) — FastAPI proxy app; also mounts the static frontend at `/`.
- [web/](web/) — Next.js 15 frontend (static export) that the backend serves at `/`.
- [client.py](client.py) — CLI client for HTTP, streaming, and direct Modal calls.
- [Dockerfile](Dockerfile) — three-stage build (Node → Python deps → runtime) producing a single image.
- [cloudrun/](cloudrun/) — Terraform module + Makefile for deploying the image to GCP Cloud Run.
- [sunflower_grpo_combined_inference.ipynb](sunflower_grpo_combined_inference.ipynb) — original reference notebook.

---

## 1. Prerequisites

```bash
uv init
uv sync
source .venv/bin/activate
uv add modal requests
uv run modal setup    # one-time auth
```

#### Output
```
https://modal.com/token-flow/tf-k381a71vs2CEqGnmHUSP48

Web authentication finished successfully!
Token is connected to the sb-modal-ws workspace.
Verifying token against https://api.modal.com
Token verified successfully!
Token written to /Users/username/.modal.toml in profile sb-modal-ws.
```

You also need a Hugging Face token with access to the gated repos:

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

(Name must be `huggingface` — that's what the app references.)

## 2. Cache the model weights into a Modal Volume (one-time)

This snapshot-downloads the base model + LoRA adapter into a persistent
`sunflower-grpo-models` volume so cold starts don't re-pull weights.

```bash
uv run modal run sunflower_grpo_vllm_modal.py::download_model
```

## 3. Smoke-test locally via Modal

Runs a few notebook prompts (Luganda, Runyankole, general Q&A) plus a streaming demo:

```bash
uv run modal run sunflower_grpo_vllm_modal.py
```

## 4. Deploy

```bash
uv run modal deploy sunflower_grpo_vllm_modal.py
```

This deploys:

- The `SunflowerVLLM` class (callable via `modal.Cls.from_name(...)`).
- A FastAPI ASGI app `web` exposing:
  - `POST /generate` — JSON `{response: "..."}`
  - `POST /generate_stream` — Server-Sent Events streaming `{delta: "..."}` chunks
  - `GET  /health`

After deploy, Modal prints a URL like
`https://<workspace>--sunflower-grpo-vllm-web.modal.run`.

## 5. Test the deployment with the client

```bash
export SUNFLOWER_URL=https://<workspace>--sunflower-grpo-vllm-web.modal.run

# Blocking HTTP call
uv run client.py http "Translate to luganda: I am watching an Arsenal game right now" 0.1

uv run client.py http "Who is Sunbird AI, what do they do?"

# Streaming SSE call
uv run client.py stream "Translate to runyakole: Good morning, how are you today?" 0.2

uv run client.py stream "Who is Sunbird AI, what do they do?"

# Direct Modal SDK call (no HTTP)
uv run client.py modal "Who is Sunbird AI, what do they do?"
```

Or with `curl`:

```bash
curl -X POST $SUNFLOWER_URL/generate \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;","temperature":0.2}'

curl -N -X POST $SUNFLOWER_URL/generate_stream \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;","temperature":0.2}'
```

## 6. Run the standalone FastAPI proxy app

[fastapi_app.py](fastapi_app.py) is a thin **HTTP proxy** sitting in front of
the Modal-hosted `web` endpoint. The Modal app already exposes
`/generate`, `/generate_stream`, and `/health` directly — the proxy is there so
we can layer **auth, rate limiting, access logging, custom domains, and
request shaping** without redeploying the inference service.

### Endpoints

| Method | Path                      | Description                                                       |
|--------|---------------------------|-------------------------------------------------------------------|
| GET    | `/health`                 | Liveness for the proxy + readiness probe of the upstream          |
| POST   | `/generate`               | Blocking generation (GRPO LoRA) → `{"response": "..."}`           |
| POST   | `/generate_stream`        | SSE stream of `{"delta": "..."}` chunks (GRPO LoRA) then `[DONE]` |
| POST   | `/generate_openai`        | Blocking generation via OpenAI-compatible Sunflower-14B server    |
| POST   | `/generate_openai_stream` | SSE stream via OpenAI-compatible server (same `{"delta": "..."}` shape) |
| POST   | `/generate_production`    | Forwards to the production Sunbird AI API for A/B comparison      |

Request body for `/generate` and `/generate_stream`:

```json
{
  "instruction": "Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
  "temperature": 0.6,
  "max_tokens": 1024
}
```

Request body for `/generate_production` (forwarded as form-encoded to
`https://api.sunbird.ai/tasks/sunflower_simple`):

```json
{
  "instruction": "translate from english to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
  "model_type": "qwen",
  "temperature": 0.1,
  "system_message": ""
}
```

### Features

- **Pure HTTP proxy** built on `httpx.AsyncClient` — no Modal SDK dependency.
- **Rate limiting** (`slowapi`) — defaults to **100 requests/minute** and
  **1000 requests/day** per client IP. Returns HTTP 429 when exceeded.
- **Access logging** with per-request `x-request-id` header and timing.
- **Streaming passthrough** of upstream SSE (no buffering).
- All defaults configurable via env vars.

### Configuration (env vars)

| Variable                 | Default                                                          | Notes |
|--------------------------|------------------------------------------------------------------|-------|
| `SUNFLOWER_UPSTREAM_URL` | `https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run`        | Modal `web` endpoint to proxy (GRPO LoRA). |
| `UPSTREAM_TIMEOUT`       | `600`                                                            | Per-request timeout (seconds). |
| `SUNBIRD_PROD_URL`       | `https://api.sunbird.ai`                                         | Base URL of the production Sunbird AI API. |
| `SUNBIRD_PROD_TOKEN`     | _(unset)_                                                        | Bearer token for the production API. Required for `/generate_production`. |
| `SUNFLOWER_OPENAI_URL`   | `https://sb-modal-ws--sunflower-14b-openai-serve.modal.run/v1`   | OpenAI-compatible vLLM upstream (must include `/v1`). |
| `VLLM_API_KEY`           | _(unset)_                                                        | Bearer token for the OpenAI-compatible server. Required if the upstream was deployed with the `vllm-api-key` Modal secret. |
| `SUNFLOWER_OPENAI_MODEL` | `Sunbird/Sunflower-14B`                                          | Model id sent to the OpenAI server. |
| `SUNFLOWER_SYSTEM_MESSAGE` | _(built-in Sunflower system prompt)_                           | Override the system message prepended to OpenAI chat requests. |
| `RATE_LIMIT_PER_MINUTE`  | `100/minute`                                                     | Per-IP per-minute limit. |
| `RATE_LIMIT_PER_DAY`     | `1000/day`                                                       | Per-IP per-day limit. |
| `RATE_LIMIT_STORAGE_URI` | `memory://`                                                      | Use `redis://...` for multi-replica deployments. |
| `LOG_LEVEL`              | `INFO`                                                           | Standard Python logging level. |

### Run locally

```bash
uv add fastapi uvicorn httpx slowapi
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Optional: custom limits + production comparison:

```bash
export RATE_LIMIT_PER_MINUTE=60/minute
export RATE_LIMIT_PER_DAY=500/day
export SUNBIRD_PROD_TOKEN=prod_access_token_here   # enables /generate_production
```

### Test it

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;","temperature":0.2}'

curl -N -X POST http://localhost:8000/generate_stream \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;","temperature":0.2}'

# Compare against production Sunbird AI API
curl -X POST http://localhost:8000/generate_production \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"translate from english to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "model_type":"qwen",
        "temperature":0.1,
        "system_message":""
      }'

# --- OpenAI-compatible Sunflower-14B upstream --------------------------------
# Translation → temperature 0.3
curl -X POST http://localhost:8000/generate_openai \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "temperature":0.3,
        "max_tokens":1024
      }' | jq

# Non-translation → temperature 0.6
curl -X POST http://localhost:8000/generate_openai \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Who is Sunbird AI, what do they do?",
        "temperature":0.6,
        "max_tokens":1024
      }' | jq

# Streaming, translation → temperature 0.3
curl -N -X POST http://localhost:8000/generate_openai_stream \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to runyankole: Good morning, how are you today?",
        "temperature":0.3,
        "max_tokens":512
      }'

# Streaming, non-translation → temperature 0.6
curl -N -X POST http://localhost:8000/generate_openai_stream \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Write a short poem about sunflowers.",
        "temperature":0.6,
        "max_tokens":512
      }'
```

Or point [client.py](client.py) at the local proxy:

```bash
export SUNFLOWER_URL=http://localhost:8000
uv run client.py http   "Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;" 0.2
uv run client.py stream "Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;" 0.2
```

### Notes on rate limiting

- The default `memory://` backend keeps counters per-process. If you run
  multiple uvicorn workers or replicas, set `RATE_LIMIT_STORAGE_URI` to a
  Redis URL (e.g. `redis://localhost:6379`) so all workers share state.
- `slowapi` uses the client IP via `get_remote_address`. Behind a reverse
  proxy / load balancer, ensure `X-Forwarded-For` is forwarded and consider
  swapping the key function to read it.

## 7. Dockerize + deploy (frontend + backend in one image) to GCP Cloud Run (Terraform)

The single container ships both the FastAPI proxy **and** a static Next.js
frontend (translation + free-chat A/B comparison UI). FastAPI mounts the
pre-built `web/out/` at `/`, so everything lives on the same origin — no
CORS, no second service.

Layout:

- [Dockerfile](Dockerfile) — three-stage: `node:22-slim` builds `web/`, `python:3.12-slim` installs deps via `uv`, runtime stage ships both.
- [web/](web/) — Next.js 15 App Router + Tailwind. Built with `BUILD_MODE=export next build` into `web/out/`.
- [cloudrun/](cloudrun/) — Terraform module (local state) + `Makefile` for build/push/apply.

### 7.0 Local development

**Integrated mode (recommended for running the app, single terminal)** —
FastAPI serves both the API and the built frontend at `http://localhost:8000/`.

```bash
make install        # one-time: npm install in web/, uv sync
make serve          # builds web/out/ then runs uvicorn --reload on :8000
```

`uvicorn --reload` auto-reloads when `fastapi_app.py` changes; re-run
`make web` (or `make serve`) after editing anything under `web/` to rebuild
the static bundle.

**Hot-reload mode (two terminals, only needed when iterating on the UI)** —
Next.js dev server on `:3000` proxies API calls to FastAPI on `:8000`.

```bash
# Terminal 1
make dev-api        # backend only on :8000

# Terminal 2
make dev-web        # next dev on :3000 (rewrites /health, /generate* to :8000)
# → http://localhost:3000
```

`next.config.ts` rewrites are gated on `BUILD_MODE !== "export"`, so they
power the dev proxy but have no effect on the production static build.

### 7.1 Prerequisites

```bash
# Tools
gcloud  --version      # >= 450
terraform version      # >= 1.6
docker  --version      # with buildx
node    --version      # >= 20 (only needed for local dev; the image build uses its own Node)
npm     --version

# Auth (one-time)
make -C cloudrun auth
```

### 7.2 App / infra configuration

Defaults, matching the request above:

```
APP=sunflower-grpo-test
PROJECT_ID=sb-gcp-project-01
REGION=europe-west1
PORT=8080
REPO=sunflower-grpo-test
TAG=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${APP}
```

Copy the example tfvars and set the production bearer token:

```bash
cp cloudrun/terraform.tfvars.example cloudrun/terraform.tfvars
# edit cloudrun/terraform.tfvars → set sunbird_prod_token
```

### 7.3 One-shot deploy

From inside [cloudrun/](cloudrun/):

```bash
make tf-init                 # first time only
make deploy                  # build+push linux/amd64 image, then terraform apply
make url                     # print the Cloud Run URL
```

`make deploy` uses the short git SHA as the image tag (and also pushes
`:latest`), and passes it to Terraform via `-var image_tag=...`.

### 7.4 Manual steps (if you'd rather not use `make deploy`)

```bash
# From repo root (so the build context sees fastapi_app.py, pyproject.toml, uv.lock)
cd modal-deploy/sunflower-grpo-inference-vllm

TAG=$(git rev-parse --short HEAD)
IMAGE=europe-west1-docker.pkg.dev/sb-gcp-project-01/sunflower-grpo-test/sunflower-grpo-test

# Build + push
docker buildx build --platform linux/amd64 \
  -t $IMAGE:$TAG -t $IMAGE:latest --push .

# Apply
cd cloudrun
terraform apply -var="image_tag=$TAG"
```

### 7.5 Test the Cloud Run service

Pick up the service URL from Terraform (or substitute the one Terraform
prints, e.g. `https://sunflower-grpo-test-<hash>-ew.a.run.app`):

```bash
export SUNFLOWER_URL=$(cd cloudrun && terraform output -raw service_url)
```

#### 7.5.1 Liveness + upstream probe

```bash
curl -sS $SUNFLOWER_URL/health | jq
```

Expected:

```json
{
  "status": "ok",
  "upstream": "https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run",
  "upstream_status": "ok"
}
```

#### 7.5.2 Blocking generation — `POST /generate`

```bash
curl -sS -X POST $SUNFLOWER_URL/generate \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "temperature":0.2,
        "max_tokens":1024
      }' | jq
```

Returns `{"response": "..."}`.

#### 7.5.3 Streaming generation — `POST /generate_stream`

`-N` disables curl output buffering so you see SSE chunks as they arrive.

```bash
curl -N -sS -X POST $SUNFLOWER_URL/generate_stream \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "temperature":0.2,
        "max_tokens":1024
      }'
```

Stream is a sequence of `data: {"delta": "..."}` lines terminated with
`data: [DONE]`.

#### 7.5.4 OpenAI-compatible Sunflower-14B — `POST /generate_openai` and `/generate_openai_stream`

Both forward to the `sunflower-14b-openai` Modal app. Translation prompts
should run cooler (`temperature: 0.3`) than free-form generation (`0.6`).

```bash
# Blocking, translation → temperature 0.3
curl -sS -X POST $SUNFLOWER_URL/generate_openai \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "temperature":0.3,
        "max_tokens":1024
      }' | jq

# Blocking, non-translation → temperature 0.6
curl -sS -X POST $SUNFLOWER_URL/generate_openai \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Who is Sunbird AI, what do they do?",
        "temperature":0.6,
        "max_tokens":1024
      }' | jq

# Streaming, translation → temperature 0.3
curl -N -sS -X POST $SUNFLOWER_URL/generate_openai_stream \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Translate to runyankole: Good morning, how are you today?",
        "temperature":0.3,
        "max_tokens":512
      }'

# Streaming, non-translation → temperature 0.6
curl -N -sS -X POST $SUNFLOWER_URL/generate_openai_stream \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"Write a short poem about sunflowers.",
        "temperature":0.6,
        "max_tokens":512
      }'
```

Stream wire format matches `/generate_stream` — `data: {"delta": "..."}` lines
terminated with `data: [DONE]` — so frontend readers written against the GRPO
stream work unchanged.

#### 7.5.5 Production comparison — `POST /generate_production`

Proxied to `https://api.sunbird.ai/tasks/sunflower_simple` using
`SUNBIRD_PROD_TOKEN` configured on the Cloud Run service.

```bash
curl -sS -X POST $SUNFLOWER_URL/generate_production \
  -H 'content-type: application/json' \
  -d '{
        "instruction":"translate from english to luganda: To promote standardisation in the planning, acquisition, implementation, delivery, support and maintenance of information technology equipment and services, to ensure uniformity in quality, adequacy and reliability of information technology usage throughout Uganda;",
        "model_type":"qwen",
        "temperature":0.1,
        "system_message":""
      }' | jq
```

Returns the production API's JSON body unchanged.

#### 7.5.6 Rate-limit sanity check

Defaults are **100 req/min** and **1000 req/day** per IP. Firing 105 quick
requests should produce a mix of `200` and `429` responses:

```bash
for i in $(seq 1 105); do
  curl -s -o /dev/null -w "%{http_code}\n" $SUNFLOWER_URL/health
done | sort | uniq -c
```

Every response also carries an `x-request-id` header from the proxy — grab
it with `-D -` to correlate with logs (`make -C cloudrun logs`).

### 7.6 Tail logs

```bash
make -C cloudrun logs
```

### 7.7 Notes

- **State** is local (`cloudrun/terraform.tfstate`). Commit nothing from
  `cloudrun/` other than `*.tf`, `terraform.tfvars.example`, the `Makefile`,
  and `.gitignore`.
- **Secrets**: `sunbird_prod_token` is passed to Cloud Run as a plain env var
  via tfvars. If you later want to rotate without re-applying, migrate it to
  Secret Manager + `env.value_source.secret_key_ref`.
- **Region**: `europe-west1` for both Artifact Registry and Cloud Run, so
  image pulls stay in-region.
- **Platform**: the image is built `--platform linux/amd64` — required when
  building from an Apple Silicon Mac.
- **Frontend build**: happens inside the Docker build's Node stage, so you
  don't need Node installed to ship. Re-runs any time `web/**` changes.
- **UI modes**: the landing page starts in **Chat** mode (free-form
  instruction → A/B against production). Clicking **Translate** swaps the
  input to From / To language pickers + text area; **Exit translation**
  returns to chat. Both modes fire `/generate_stream` (GRPO, streaming) and
  `/generate_production` (blocking) in parallel; two panels show results
  side-by-side with per-panel latency.

## 8. Configuration knobs

Edit constants at the top of [sunflower_grpo_vllm_modal.py](sunflower_grpo_vllm_modal.py):

| Constant         | Default          | Notes |
|------------------|------------------|-------|
| `GPU_TYPE`       | `A100-40GB`      | Alternatives: `A100-80GB`, `L40S` (48 GB), `H100`. |
| `MAX_MODEL_LEN`  | `1024`           | Increase for longer context (more KV cache). |
| `MAX_LORA_RANK`  | `32`             | Must be ≥ adapter rank. |
| `scaledown_window` | `300` (5 min)  | Idle time before container spins down. |
| `@modal.concurrent(max_inputs=8)` | `8` | vLLM batches concurrent requests on one warm replica. |

## 9. Troubleshooting

- **OOM at engine init** — drop `gpu_memory_utilization` to `0.85` or move to `A100-80GB` / `L40S`.
- **Gated repo 401** — confirm the `huggingface` Modal secret has a token with access to the two `jq/...` repos.
- **Streaming hangs** — use `curl -N` (no buffering) and ensure your client reads `text/event-stream` line-by-line.
- **Re-download weights** — `modal volume rm sunflower-grpo-models && modal run sunflower_grpo_vllm_modal.py::download_model`.
