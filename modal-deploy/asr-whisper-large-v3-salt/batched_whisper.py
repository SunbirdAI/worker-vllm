# # Fast Whisper inference using dynamic batching

# In this example, we demonstrate how to run [dynamically batched inference](https://modal.com/docs/guide/dynamic-batching)
# for OpenAI's speech recognition model, [Whisper](https://openai.com/index/whisper/), on Modal.
# Batching multiple audio samples together or batching chunks of a single audio sample can help to achieve a 2.8x increase
# in inference throughput on an A10G!

# We will be running the [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) model.
# To run [any of the other HuggingFace Whisper models](https://huggingface.co/models?search=openai/whisper),
# simply replace the `MODEL_NAME` and `MODEL_REVISION` variables.

# ## Setup

# Let's start by importing the Modal client and defining the model that we want to serve.


from typing import Optional

import modal
from fastapi import Request

MODEL_NAME = "Sunbird/asr-whisper-large-v3-salt"

# cache model weights with Modal Volumes
HF_CACHE_DIR = "/root/.cache/huggingface"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# ## Define a container image

# We’ll start with Modal's baseline `debian_slim` image and install the relevant libraries.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "huggingface-hub==0.36.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "torchaudio==2.5.1",
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": HF_CACHE_DIR})
)

app = modal.App(
    "asr-whisper-large-v3-salt",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-read")],
    volumes={HF_CACHE_DIR: hf_cache_vol},
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
    gpu="a10g",  # Try using an A100 or H100 if you've got a large model or need big batches!
    max_containers=10,  # default max GPUs for Modal's free tier
    scaledown_window=60 * 3,
)
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import pipeline

        # Create a pipeline for preprocessing and transcribing speech data
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device="cuda",
            torch_dtype=torch.float16,
        )

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

    @modal.fastapi_endpoint(docs=True, method="POST")
    async def transcribe(self, request: Request):
        """
        Web endpoint that accepts audio bytes and returns the transcription.
        """
        import time

        data = await request.body()
        generate_kwargs = {
            # "language": 'English', 
            "task": "transcribe",
            "num_beams": 1,
            "return_timestamps": True,
        }

        start = time.monotonic_ns()
        transcriptions = self.pipeline(
            [data], 
            batch_size=1, 
            generate_kwargs=generate_kwargs,
        )
        end = time.monotonic_ns()
        print(
            f"Transcribed in {round((end - start) / 1e9, 2)}s"
        )
        
        return {"text": transcriptions[0]["text"]}


# ## Transcribe a dataset

# In this example, we use the [librispeech_asr_dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
# from Hugging Face's Datasets library to test the model.

# We use [`map.aio`](https://modal.com/docs/reference/modal.Function#map) to asynchronously map over the audio files.
# This allows us to invoke the batched transcription method on each audio sample in parallel.


@app.function()
async def transcribe_hf_dataset(dataset_name):
    from datasets import load_dataset

    print("📂 Loading dataset", dataset_name)
    ds = load_dataset(dataset_name, "multispeaker-eng", split="test")
    print("📂 Dataset loaded")
    batched_whisper = Model()
    print("📣 Sending data for transcription")
    async for transcription in batched_whisper.transcribe.map.aio(ds["audio"]):
        yield transcription


# ## Run the model

# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the transcription. You can run this locally with `modal run batched_whisper.py`.


@app.local_entrypoint()
async def main(dataset_name: Optional[str] = None):
    if dataset_name is None:
        dataset_name = "Sunbird/salt"
    for result in transcribe_hf_dataset.remote_gen(dataset_name):
        print(result["text"])
