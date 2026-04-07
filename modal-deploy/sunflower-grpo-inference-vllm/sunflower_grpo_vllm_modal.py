"""
Modal deployment of Sunflower 14B GRPO Combined (base + LoRA adapter) using vLLM.

Base model:    jq/sunflower-14b-bs64-lr1e-4-250919
LoRA adapter:  jq/sunflower-14b-grpo-combined

Endpoints exposed:
  - POST /generate         -> JSON {response: "..."}
  - POST /generate_stream  -> text/event-stream (SSE), token-by-token
Also exposes Modal class methods callable via `.remote()` from Python clients.

Deploy:
    modal run  sunflower_grpo_vllm_modal.py::download_model   # one-time weight cache
    modal deploy sunflower_grpo_vllm_modal.py
"""

from __future__ import annotations

import json
import os

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "sunflower-grpo-vllm"
BASE_MODEL_REPO = "jq/sunflower-14b-bs64-lr1e-4-250919"
LORA_REPO = "jq/sunflower-14b-grpo-combined"

MODELS_DIR = "/models"
BASE_MODEL_DIR = f"{MODELS_DIR}/base"
LORA_DIR = f"{MODELS_DIR}/lora"

MAX_MODEL_LEN = 1024
MAX_LORA_RANK = 32
GPU_TYPE = "A100-40GB"  # alternatives: "A100-80GB", "L40S" (48GB), "H100"

# FAST_BOOT=True  -> enforce_eager=True  (faster cold start, slower steady-state)
# FAST_BOOT=False -> enforce_eager=False (slower cold start, CUDA graphs + torch.compile)
FAST_BOOT = True

SYSTEM_MESSAGE = (
    "You are Sunflower, a helpful assistant made by Sunbird AI who understands all "
    "Ugandan languages. You specialise in accurate translations, explanations, "
    "summaries and other language tasks."
)

# ---------------------------------------------------------------------------
# Modal image, volume, secrets
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]>=0.34.0,<1.0",
        "transformers>=4.55.2,<5.0",
        "fastapi[standard]==0.115.4",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
        }
    )
)

models_volume = modal.Volume.from_name(
    "sunflower-grpo-models", create_if_missing=True
)
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# One-time weight download into the volume
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    volumes={MODELS_DIR: models_volume},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def download_model() -> None:
    """Snapshot-download the base model and LoRA adapter into the persistent volume."""
    from huggingface_hub import snapshot_download

    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    os.makedirs(LORA_DIR, exist_ok=True)

    print(f"Downloading base model {BASE_MODEL_REPO} -> {BASE_MODEL_DIR}")
    snapshot_download(
        repo_id=BASE_MODEL_REPO,
        local_dir=BASE_MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"Downloading LoRA adapter {LORA_REPO} -> {LORA_DIR}")
    snapshot_download(
        repo_id=LORA_REPO,
        local_dir=LORA_DIR,
        token=os.environ.get("HF_TOKEN"),
    )

    models_volume.commit()
    print("Done. Weights cached in volume `sunflower-grpo-models`.")


# ---------------------------------------------------------------------------
# vLLM serving class
# ---------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={MODELS_DIR: models_volume},
    secrets=[hf_secret],
    scaledown_window=300,
    timeout=60 * 30,
)
@modal.concurrent(max_inputs=8)
class SunflowerVLLM:
    @modal.enter()
    def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.lora.request import LoRARequest

        print("Loading vLLM AsyncLLM engine...")
        engine_args = AsyncEngineArgs(
            model=BASE_MODEL_DIR,
            enable_lora=True,
            max_lora_rank=MAX_LORA_RANK,
            max_loras=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            enforce_eager=FAST_BOOT,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
        self.lora_request = LoRARequest(
            lora_name="sunflower-grpo",
            lora_int_id=1,
            lora_path=LORA_DIR,
        )
        print("vLLM engine ready.")

    # -- helpers ----------------------------------------------------------
    def _build_prompt(self, instruction: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def _sampling(self, temperature: float, max_tokens: int):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=max_tokens,
        )

    # -- non-streaming method --------------------------------------------
    @modal.method()
    async def generate(
        self,
        instruction: str,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        prompt = self._build_prompt(instruction)
        sampling = self._sampling(temperature, max_tokens)
        request_id = f"req-{os.urandom(4).hex()}"
        final_text = ""
        async for out in self.engine.generate(
            prompt, sampling, request_id, lora_request=self.lora_request
        ):
            final_text = out.outputs[0].text
        return final_text

    # -- streaming method (yields delta strings) -------------------------
    @modal.method()
    async def generate_stream(
        self,
        instruction: str,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ):
        prompt = self._build_prompt(instruction)
        sampling = self._sampling(temperature, max_tokens)
        request_id = f"req-{os.urandom(4).hex()}"
        previous = ""
        async for out in self.engine.generate(
            prompt, sampling, request_id, lora_request=self.lora_request
        ):
            text = out.outputs[0].text
            delta = text[len(previous):]
            previous = text
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# HTTP endpoints (FastAPI via Modal)
# ---------------------------------------------------------------------------
web_image = vllm_image


@app.function(image=web_image, secrets=[hf_secret])
@modal.asgi_app()
def web():
    from fastapi import Body, FastAPI
    from fastapi.responses import StreamingResponse

    api = FastAPI(title="Sunflower GRPO vLLM")

    @api.post("/generate")
    async def generate(
        instruction: str = Body(...),
        temperature: float = Body(0.6),
        max_tokens: int = Body(1024),
    ):
        text = await SunflowerVLLM().generate.remote.aio(
            instruction, temperature, max_tokens
        )
        return {"response": text}

    @api.post("/generate_stream")
    async def generate_stream(
        instruction: str = Body(...),
        temperature: float = Body(0.6),
        max_tokens: int = Body(1024),
    ):
        async def event_source():
            async for delta in SunflowerVLLM().generate_stream.remote_gen.aio(
                instruction, temperature, max_tokens
            ):
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @api.get("/health")
    def health():
        return {"status": "ok"}

    return api


# ---------------------------------------------------------------------------
# Local entrypoint: smoke-test parity with the notebook
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    prompts = [
        ("Translate to luganda: I am watching an Arsenal game right now", 0.1),
        (
            "Translate to runyakole: The winner of Sunday's Carabao Cup Final will "
            "gain qualification to the Europa Conference League play-off round next season.",
            0.1,
        ),
        ("Who is Sunbird AI, what do they do?", 0.6),
    ]
    cls = SunflowerVLLM()
    for instr, temp in prompts:
        print(f"\n>>> {instr}")
        print(cls.generate.remote(instr, temp))

    print("\n--- streaming demo ---")
    for delta in cls.generate_stream.remote_gen(
        "Translate to luganda: Good morning, how are you today?", 0.2
    ):
        print(delta, end="", flush=True)
    print()
