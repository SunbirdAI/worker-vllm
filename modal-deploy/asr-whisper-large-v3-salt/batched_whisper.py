# # Fast Whisper inference with vLLM
#
# Deploy:
# ```shell
# modal deploy batched_whisper.py
# ```
#
# Query the endpoint:
# ```shell
# curl -X POST "https://thu-58148--asr-whisper-large-v3-salt-model-transcribe.modal.run" \
#   --data-binary "@../../sunflower-ultravox-vllm/audios/context_eng_6.wav" \
#   -H "Content-Type: application/octet-stream"
# ```

import io
from typing import Optional
import modal
from fastapi import Request

MODEL_NAME = "Sunbird/asr-whisper-large-v3-salt"

# Cache model weights with Modal Volumes
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Define a container image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "torch==2.9.0",
        "transformers==4.56.0",
        "huggingface-hub==0.36.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "torchaudio==2.9.0",
        "fastapi[standard]",
        "python-multipart",
        "vllm==0.12.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": HF_CACHE_DIR})
)

# Import libraries within image context
with image.imports():
    import time
    import librosa
    from vllm import LLM, SamplingParams

app = modal.App(
    "asr-whisper-large-v3-salt",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-read")],
    volumes={HF_CACHE_DIR: hf_cache_vol, VLLM_CACHE_DIR: vllm_cache_vol},
)

# ## Caching the model weights

# We'll define a function to download the model and cache it in a volume.
# You can `modal run batched_whisper.py::download_model` against this function prior to deploying the App.


@app.function()
def download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    snapshot_download(
        MODEL_NAME,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


# ## The model class

# The inference function is best represented using Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).

# We define a `@modal.enter` method to load the model when the container starts, before it picks up any inputs.
# The weights will be loaded from the Hugging Face cache volume so that we don't need to download them when
# we start a new container. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).



@app.cls(
    gpu="A10",  # Try using an A100 or H100 if you've got a large model or need big batches!
    max_containers=1,  # default max GPUs for Modal's free tier
    scaledown_window=60 * 2,
    # enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=32)
class Model:
    @modal.enter()
    def load_model(self):
        # import torch
        # from transformers import pipeline

        # # Create a pipeline for preprocessing and transcribing speech data
        # self.pipeline = pipeline(
        #     "automatic-speech-recognition",
        #     model=MODEL_NAME,
        #     device="cuda",
        #     torch_dtype=torch.float16,
        # )
        print("Loading Whisper model with vLLM...")
        self.model = LLM(
            MODEL_NAME,
            enforce_eager=False,
            gpu_memory_utilization=0.8,
            max_model_len=448,
            max_num_seqs=32,
            limit_mm_per_prompt={"audio": 1},
        )
        print("✅ Model loaded successfully!")

    # @modal.batched(max_batch_size=64, wait_ms=1000)
    # def transcribe(self, audio_samples):
    #     import time

    #     generate_kwargs = {
    #         "language": 'English', 
    #         "task": "transcribe",
    #         "num_beams": 1,
    #     }

    #     start = time.monotonic_ns()
    #     print(f"Transcribing {len(audio_samples)} audio samples")
    #     transcriptions = self.pipeline(
    #         audio_samples, 
    #         batch_size=len(audio_samples), 
    #         generate_kwargs=generate_kwargs
    #     )
    #     end = time.monotonic_ns()
    #     print(
    #         f"Transcribed {len(audio_samples)} samples in {round((end - start) / 1e9, 2)}s"
    #     )
    #     return transcriptions

    @modal.method()
    async def transcribe(self, audio):
        """
        Web endpoint that accepts audio bytes and returns the transcription.
        """
        
        # Load audio from bytes
        
        start = time.monotonic_ns()
        
        # Prepare input with multimodal data
        inputs = {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": [(audio["array"], audio["sampling_rate"])]
            }
        }
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=448)
        
        # Use vLLM generate
        outputs = self.model.generate([inputs], sampling_params=sampling_params)
        
        end = time.monotonic_ns()
        print(f"Transcribed in {round((end - start) / 1e9, 2)}s")

        return {"text": outputs[0].outputs[0].text}


# ## Transcribe a dataset

# We use [`map.aio`](https://modal.com/docs/reference/modal.Function#map) to asynchronously map over the audio files.
# This allows us to invoke the batched transcription method on each audio sample in parallel.

# invoke with web url, e.g. https://thu-58148--asr-whisper-large-v3-salt-transcribe-hf-dataset.modal.run
# then check the logs in https://modal.com/apps

@app.function()
@modal.fastapi_endpoint()
async def transcribe_hf_dataset(dataset_name="Sunbird/salt", subset="multispeaker-eng", split="test"):
    from datasets import load_dataset

    print("📂 Loading dataset", dataset_name)
    ds = load_dataset(dataset_name, subset, split=split)
    print("📂 Dataset loaded")
    batched_whisper = Model()
    print("📣 Sending data for transcription")
    import time
    map_start = time.monotonic_ns()
    async for transcription in batched_whisper.transcribe.map.aio(ds["audio"]):
        print(transcription["text"])
        # yield transcription
    map_end = time.monotonic_ns()
    print(f"⏱️ Total map transcription time: {round((map_end - map_start) / 1e9, 2)}s")


# ## Run the model

# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the transcription. You can run this locally with `modal run batched_whisper.py`.


# @app.local_entrypoint()
# async def main(dataset_name: Optional[str] = None):
#     if dataset_name is None:
#         dataset_name = "Sunbird/salt"
#     for result in transcribe_hf_dataset.remote_gen(dataset_name):
#         print(result["text"])
