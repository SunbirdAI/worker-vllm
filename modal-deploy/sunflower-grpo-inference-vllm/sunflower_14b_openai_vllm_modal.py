"""
Modal deployment of Sunbird/Sunflower-14B as an OpenAI-compatible vLLM server.

Serves the standard OpenAI REST surface:
  - GET  /v1/models
  - POST /v1/chat/completions        (supports stream=true for SSE)
  - POST /v1/completions             (supports stream=true for SSE)
  - GET  /health

Deploy:
    # 1. Create required secrets (one-time)
    modal secret create huggingface-secret HF_TOKEN=<your_token>
    modal secret create vllm-api-key VLLM_API_KEY=<pick_a_key>   # optional, see AUTH below

    # 2. Deploy
    modal deploy sunflower_14b_openai_vllm_modal.py

Test:
    export BASE="https://<workspace>--sunflower-14b-openai-serve.modal.run"
    export KEY="<your VLLM_API_KEY>"

    curl -s "$BASE/v1/chat/completions" \
      -H "Authorization: Bearer $KEY" \
      -H "content-type: application/json" \
      -d '{
        "model": "Sunbird/Sunflower-14B",
        "messages": [{"role":"user","content":"Translate to luganda: Good morning"}],
        "stream": false
      }' | jq .

Streaming (SSE):
    curl -N "$BASE/v1/chat/completions" \
      -H "Authorization: Bearer $KEY" \
      -H "content-type: application/json" \
      -d '{
        "model": "Sunbird/Sunflower-14B",
        "messages": [{"role":"user","content":"Tell me a short story"}],
        "stream": true
      }'

Python client:
    from openai import OpenAI
    client = OpenAI(base_url=f"{BASE}/v1", api_key=KEY)
    resp = client.chat.completions.create(
        model="Sunbird/Sunflower-14B",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    for chunk in resp:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

AUTH:
    If the `vllm-api-key` secret exists and defines VLLM_API_KEY, the server
    requires `Authorization: Bearer $VLLM_API_KEY` on every request.
    To run without auth, remove the secret from the `secrets=` list below
    (note: the endpoint will then be publicly reachable).
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "sunflower-14b-openai"
MODEL_NAME = "Sunbird/Sunflower-14B"
MODEL_REVISION = "main"
SERVED_MODEL_ALIAS = "sunflower-14b"   # extra alias clients may send as `model=`

VLLM_PORT = 8000
MAX_MODEL_LEN = 4096
GPU_SPEC = "L40S:1"   # 48GB. Alternatives: "H100:1" (80GB), "A100-80GB:1" (80GB)

# FAST_BOOT=True  -> --enforce-eager (faster cold start, slower steady-state)
# FAST_BOOT=False -> --no-enforce-eager (slower cold start, CUDA graphs + torch.compile)
FAST_BOOT = True

MINUTE = 60

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        "transformers>=4.55.2,<5.0",
        "hf-transfer>=0.1.8",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
        }
    )
)

# ---------------------------------------------------------------------------
# Volumes for model + compilation caches (HF cache path + vLLM compile cache)
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name(
    "sunflower-14b-hf-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name(
    "sunflower-14b-vllm-cache", create_if_missing=True
)

hf_secret = modal.Secret.from_name("huggingface-secret")
api_key_secret = modal.Secret.from_name("vllm-api-key")

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# OpenAI-compatible vLLM server
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret, api_key_secret],
    scaledown_window=15 * MINUTE,
    timeout=24 * 60 * MINUTE,
    min_containers=0,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTE)
def serve() -> None:
    """Boot vLLM's built-in OpenAI-compatible API server as a subprocess."""
    import os
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        SERVED_MODEL_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.90",
        "--uvicorn-log-level",
        "info",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture.
    # Default is --no-enforce-eager. See --compilation-config for tighter control.
    cmd.append("--enforce-eager" if FAST_BOOT else "--no-enforce-eager")

    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        cmd.extend(["--api-key", api_key])

    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd)
