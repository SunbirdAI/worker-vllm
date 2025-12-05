# Run OpenAI-compatible LLM inference with vLLM
# see documentations in: https://modal.com/docs/examples/vllm_inference#build-a-vllm-engine-and-serve-it

import json
from typing import Any

import aiohttp
import modal

# define container image
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm[audio]==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# Download the model weights

MODEL_NAME = "huwenjie333/sunflower32b-ultravox-251114-3"

# cache model weights with Modal Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Trading off fast boots and token generation performance
FAST_BOOT = True

# Build a vLLM engine and serve it
app = modal.App("Sunflower32b-Ultravox")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000

@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=2 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-read")],
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--max-model-len 4096"
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)

# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_inference.py
# ```

# ## Testing the server
# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# ```bash
# modal run vllm_inference.py
# ```

@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=False):
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=1 * MINUTES
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()
