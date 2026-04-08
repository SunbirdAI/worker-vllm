# Sunflower 14B GRPO — Modal + vLLM Deployment

Serves the **`jq/sunflower-14b-bs64-lr1e-4-250919`** base model with the
**`jq/sunflower-14b-grpo-combined`** LoRA adapter on [Modal](https://modal.com)
using **vLLM**, exposing both blocking and streaming HTTP endpoints.

Files:

- [sunflower_grpo_vllm_modal.py](sunflower_grpo_vllm_modal.py) — Modal app (image, volume, vLLM class, FastAPI endpoints).
- [fastapi_app.py](fastapi_app.py) — standalone FastAPI app that proxies to the deployed Modal class.
- [client.py](client.py) — CLI client for HTTP, streaming, and direct Modal calls.
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
  -d '{"instruction":"Translate to luganda: Good morning","temperature":0.2}'

curl -N -X POST $SUNFLOWER_URL/generate_stream \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: Good morning","temperature":0.2}'
```

## 6. Run the standalone FastAPI proxy app

[fastapi_app.py](fastapi_app.py) is a thin **HTTP proxy** sitting in front of
the Modal-hosted `web` endpoint. The Modal app already exposes
`/generate`, `/generate_stream`, and `/health` directly — the proxy is there so
we can layer **auth, rate limiting, access logging, custom domains, and
request shaping** without redeploying the inference service.

### Endpoints

| Method | Path               | Description                                                       |
|--------|--------------------|-------------------------------------------------------------------|
| GET    | `/health`          | Liveness for the proxy + readiness probe of the upstream          |
| POST   | `/generate`        | Blocking generation → `{"response": "..."}`                       |
| POST   | `/generate_stream` | Server-Sent Events stream of `{"delta": "..."}` chunks then `[DONE]` |

Request body for `/generate` and `/generate_stream`:

```json
{
  "instruction": "Translate to luganda: Good morning",
  "temperature": 0.6,
  "max_tokens": 1024
}
```

### Features

- **Pure HTTP proxy** built on `httpx.AsyncClient` — no Modal SDK dependency.
- **Optional bearer auth** via `SUNFLOWER_API_KEY` env var (no-op if unset).
- **Rate limiting** (`slowapi`) — defaults to **100 requests/minute** and
  **1000 requests/day** per client IP. Returns HTTP 429 when exceeded.
- **Access logging** with per-request `x-request-id` header and timing.
- **Streaming passthrough** of upstream SSE (no buffering).
- All defaults configurable via env vars.

### Configuration (env vars)

| Variable                 | Default                                                          | Notes |
|--------------------------|------------------------------------------------------------------|-------|
| `SUNFLOWER_UPSTREAM_URL` | `https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run`        | Modal `web` endpoint to proxy. |
| `SUNFLOWER_API_KEY`      | _(unset)_                                                        | If set, clients must send `Authorization: Bearer <key>`. |
| `UPSTREAM_TIMEOUT`       | `600`                                                            | Per-request timeout (seconds). |
| `RATE_LIMIT_PER_MINUTE`  | `100/minute`                                                     | Per-IP per-minute limit. |
| `RATE_LIMIT_PER_DAY`     | `1000/day`                                                       | Per-IP per-day limit. |
| `RATE_LIMIT_STORAGE_URI` | `memory://`                                                      | Use `redis://...` for multi-replica deployments. |
| `LOG_LEVEL`              | `INFO`                                                           | Standard Python logging level. |

### Run locally

```bash
uv add fastapi uvicorn httpx slowapi
uv run uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

Optional auth + custom limits:

```bash
export SUNFLOWER_API_KEY=secret123
export RATE_LIMIT_PER_MINUTE=60/minute
export RATE_LIMIT_PER_DAY=500/day
```

### Test it

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/generate \
  -H 'authorization: Bearer secret123' \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: Good morning","temperature":0.2}'

curl -N -X POST http://localhost:8000/generate_stream \
  -H 'authorization: Bearer secret123' \
  -H 'content-type: application/json' \
  -d '{"instruction":"Translate to luganda: Good morning","temperature":0.2}'
```

Or point [client.py](client.py) at the local proxy:

```bash
export SUNFLOWER_URL=http://localhost:8000
uv run client.py http   "Translate to luganda: Good morning" 0.2
uv run client.py stream "Translate to luganda: Good morning" 0.2
```

### Notes on rate limiting

- The default `memory://` backend keeps counters per-process. If you run
  multiple uvicorn workers or replicas, set `RATE_LIMIT_STORAGE_URI` to a
  Redis URL (e.g. `redis://localhost:6379`) so all workers share state.
- `slowapi` uses the client IP via `get_remote_address`. Behind a reverse
  proxy / load balancer, ensure `X-Forwarded-For` is forwarded and consider
  swapping the key function to read it.

## 7. Configuration knobs

Edit constants at the top of [sunflower_grpo_vllm_modal.py](sunflower_grpo_vllm_modal.py):

| Constant         | Default          | Notes |
|------------------|------------------|-------|
| `GPU_TYPE`       | `A100-40GB`      | Alternatives: `A100-80GB`, `L40S` (48 GB), `H100`. |
| `MAX_MODEL_LEN`  | `1024`           | Increase for longer context (more KV cache). |
| `MAX_LORA_RANK`  | `32`             | Must be ≥ adapter rank. |
| `scaledown_window` | `300` (5 min)  | Idle time before container spins down. |
| `@modal.concurrent(max_inputs=8)` | `8` | vLLM batches concurrent requests on one warm replica. |

## 8. Troubleshooting

- **OOM at engine init** — drop `gpu_memory_utilization` to `0.85` or move to `A100-80GB` / `L40S`.
- **Gated repo 401** — confirm the `huggingface` Modal secret has a token with access to the two `jq/...` repos.
- **Streaming hangs** — use `curl -N` (no buffering) and ensure your client reads `text/event-stream` line-by-line.
- **Re-download weights** — `modal volume rm sunflower-grpo-models && modal run sunflower_grpo_vllm_modal.py::download_model`.
