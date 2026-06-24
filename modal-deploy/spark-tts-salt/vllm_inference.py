# Deploy the Spark-TTS-Salt model with vLLM:
#
# ```shell
# modal deploy vllm_inference.py
# ```
#
# And query the endpoint with:
#
# ```shell
# curl -X POST --get "https://sb-modal-ws--spark-tts-salt-sparktts-generate.modal.run" \
#   --data-urlencode "text=I am a nurse who takes care of many people who have cancer." \
#   --data-urlencode "speaker_id=248" \
#   --output output.wav
# ```
#
# You'll receive a WAV file named `output.wav` containing the generated audio.

import io
import modal
from typing import  List

# ## Define a container image
# We start with Modal's baseline `debian_slim` image and install the required packages.

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "fastapi[standard]",
        "einx",
        "einops",
        "soundfile",
        "numpy",
        "torch",
        "librosa",
        "vllm==0.12.0",
        "omegaconf",
        "huggingface_hub",
    )
    .run_commands("git clone https://github.com/SparkAudio/Spark-TTS /root/Spark-TTS")
    .env({"PYTHONPATH": "/root/Spark-TTS"})
)
app = modal.App("spark-tts-salt-job-queue", image=image)

# Import the required libraries within the image context to ensure they're available
# when the container runs. This includes audio processing and the TTS model itself.

with image.imports():
    import re
    import numpy as np
    import torch
    import soundfile as sf
    from typing import List
    import uuid
    import asyncio
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from huggingface_hub import snapshot_download
    from fastapi import Response
    from sparktts.models.audio_tokenizer import BiCodecTokenizer
    import time

# dictionary to hold job status
job_status = modal.Dict.from_name("spark-tts-job-status", create_if_missing=True)

# cache model weights with Modal Volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# cache some of vLLM's compilation artifacts in a Modal Volume.
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

# ## The TTS model class

# The TTS service is implemented using Modal's class syntax with GPU acceleration.

# - `scaledown_window=60 * 3`: Keep containers alive for 3 minutes after last request
# - `enable_memory_snapshot=True`: Enable [memory snapshots](https://modal.com/docs/guide/memory-snapshot) to optimize cold boot times
# - `@modal.concurrent(max_inputs=10)`: Allow up to 10 concurrent requests per container

@app.cls(
    gpu="L4",
    max_containers=10,
    scaledown_window=60 * 5,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("huggingface-read")],
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    }
)
@modal.concurrent(max_inputs=100)
class SparkTTS:
    # 241: Acholi (female)
    # 242: Ateso (female)
    # 243: Runyankore (female)
    # 245: Lugbara (female)
    # 246: Swahili (male)
    # 248: Luganda (female)
    GLOBAL_IDS_BY_SPEAKER = {
        241: [1755, 1265, 184, 3545, 2718, 2405, 3237, 1360, 3621, 1850, 37, 3382, 736,
            3380, 3131, 2036, 244, 2128, 254, 2550, 3181, 764, 1277, 502, 2941, 1993,
            3556, 1428, 3505, 3245, 3506, 1540],
        242: [1367, 1522, 308, 4061, 1449, 2468, 2193, 1349, 3458, 2339, 1651, 3174,
            501, 3364, 3194, 2041, 442, 1061, 502, 2234, 2397, 358, 3829, 2490, 2031,
            1002, 3548, 586, 3445, 1419, 4093, 2908],
        243: [2051, 242, 2684, 4062, 2654, 2252, 353, 3657, 2759, 3254, 1649, 3366,
            1017, 3600, 3131, 3813, 1535, 1595, 1059, 237, 2158, 1174, 4085, 2174,
            3791, 990, 3274, 2693, 3829, 2271, 2650, 1689],
        245: [2031, 2545, 116, 4060, 746, 1385, 3301, 1312, 3638, 1846, 85, 3190, 1016,
            3384, 3134, 954, 244, 1104, 235, 2549, 3357, 508, 1278, 1974, 2621, 1896,
            3812, 2185, 3061, 2941, 1187, 5],
        246: [1811, 1138, 2873, 3309, 2639, 723, 3363, 974, 1612, 2531, 1769, 3376,
            933, 3848, 3195, 2180, 2359, 1275, 3493, 3260, 2279, 3715, 3508, 2433,
            4082, 1087, 3545, 1449, 160, 3531, 2908, 2094],
        248: [2559, 1523, 440, 3789, 1438, 373, 2212, 1248, 3369, 1847, 36, 3126, 480,
            3380, 3133, 2041, 248, 2384, 730, 2554, 3182, 1785, 1277, 1013, 2425,
            1932, 3560, 1177, 2736, 2430, 2722, 261]
    }

    @modal.enter()
    def load(self):
        print("Loading Spark TTS model...")
        engine_args = AsyncEngineArgs(
            model="Sunbird/spark-tts-salt",
            enforce_eager=False,
            gpu_memory_utilization=0.8,
            max_num_seqs=100,
        ) # Leave some VRAM for the audio tokeniser
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        print("✅ Model loaded successfully!")

        # Download tokenizer model files
        model_base_repo = "unsloth/Spark-TTS-0.5B"
        print(f"Downloading tokenizer files from {model_base_repo}...")
        snapshot_download(
            repo_id=model_base_repo,
            local_dir=HF_CACHE_DIR,
            ignore_patterns=["*LLM*"],  # Skip LLM files, we only need tokenizer
        )
        print(f"✅ Tokenizer files downloaded to {HF_CACHE_DIR}")

        # Initialize the audio tokenizer
        print("Initializing audio tokenizer...")
        self.audio_tokenizer = BiCodecTokenizer(HF_CACHE_DIR)
        self.audio_tokenizer.model.to('cuda')
        print("✅ Audio tokenizer initialized!")  

    @modal.method()
    async def generate(self, text: str, speaker_id: int = 241, temperature: float = 0.6):
        # Mark as processing now that the model is loaded and we have started the method
        await job_status.put.aio(modal.current_function_call_id(), "processing")

        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"input text: {text}")
        print(f"speaker_id: {speaker_id}")
        print(f"temperature: {temperature}")

        texts = self.chunk_text_simple(text)
        texts = [t.strip() for t in texts if len(t.strip()) > 0]

        sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)

        global_tokens = self.GLOBAL_IDS_BY_SPEAKER[speaker_id]

        prompts = []
        for text in texts:
            prompt = f"<|task_tts|><|start_content|>{speaker_id}: {text}<|end_content|><|start_global_token|>"
            prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_tokens]) + '<|end_global_token|><|start_semantic_token|>'
            prompts.append(prompt)

        gen_start = time.time()
        
        async def generate_chunk(prompt):
            request_id = str(uuid.uuid4())
            results_generator = self.model.generate(prompt, sampling_params, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            return final_output

        outputs = await asyncio.gather(*[generate_chunk(p) for p in prompts])
        print(f"Model generation time: {time.time() - gen_start:.2f}s")

        decode_start = time.time()
        speech_segments = []

        for i in range(len(outputs)):
            predicted_tokens = outputs[i].outputs[0].text
            semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
            if not semantic_matches:
                raise ValueError("No semantic tokens found in the generated output.")

            pred_semantic_ids = (
                torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
            )

            pred_global_ids = torch.Tensor([global_tokens]).long()

            wav_np = await asyncio.to_thread(
                self.audio_tokenizer.detokenize,
                pred_global_ids.to(device), pred_semantic_ids.to(device)
            )
            speech_segments.append(wav_np)

        result_wav = np.concatenate(speech_segments)
        print(f"Audio decoding time: {time.time() - decode_start:.2f}s")

        save_start = time.time()
        # Create an in-memory buffer to store the WAV file
        buffer = io.BytesIO()

        # Save the generated audio to the buffer in WAV format
        # Uses the model's sample rate and WAV format
        sf.write(buffer, result_wav, self.audio_tokenizer.config["sample_rate"], format='WAV')

        # Reset buffer position to the beginning for reading
        buffer.seek(0)
        print(f"Audio saving time: {time.time() - save_start:.2f}s")

        print(f"Total generation time: {time.time() - start_time:.2f}s")
        # Return raw audio bytes
        return buffer.getvalue()

    def chunk_text_simple(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Recommended for TTS - provides maximum control with one sentence per chunk.
        
        Args:
            text: The input string to chunk
        
        Returns:
            List of individual sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


@app.function(min_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response

    web_app = FastAPI(title="Spark-TTS Job Queue API")

    @web_app.post("/submit")
    async def submit_job(text: str, speaker_id: int = 241, temperature: float = 0.6):
        # We spawn the workload and immediately return the corresponding ID
        call = await SparkTTS().generate.spawn.aio(text, speaker_id, temperature)
        return {"call_id": call.object_id}

    @web_app.get("/result/{call_id}")
    async def get_job_result(call_id: str):
        function_call = modal.FunctionCall.from_id(call_id)
        
        try:
            # We must set a 0-second timeout to check status instantly without blocking the server
            result_bytes = await function_call.get.aio(timeout=0)
            # The job finished! Remove status from dict and return the WAV response.
            if await job_status.contains.aio(call_id):
                await job_status.pop.aio(call_id)

            return Response(
                content=result_bytes,
                media_type="audio/wav",
            )
            
        except modal.exception.OutputExpiredError:
            # The job results are gone from Modal
            return JSONResponse(content={"error": "Job expired or not found."}, status_code=404)
        except TimeoutError:
            # Job is still running. Let's find out if it is loading the model or generating audio.
            status = await job_status.get.aio(call_id, "loading")
            return JSONResponse(content={"status": status}, status_code=202)

    return web_app
