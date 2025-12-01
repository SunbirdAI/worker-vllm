from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
from natsort import natsorted
from openai import OpenAI
import base64
import os
from dotenv import load_dotenv
import runpod
from pydantic import BaseModel
from typing import List, Optional
import httpx

load_dotenv()

app = FastAPI()

# Initialize OpenAI client (configure with your base_url and api_key)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
VLLM_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
print(f"Using Runpod endpoint URL: {url}")


runpod_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", url),
    api_key=os.getenv("RUNPOD_API_KEY", "your-api-key")
)

# VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://69.63.236.188:26331")
# runpod_client = OpenAI(base_url=f"{VLLM_BASE_URL}/v1", api_key="dummy")



class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str
    data: List[ModelInfo]


def list_files_sorted(directory):
    """
    List all files in a directory and return their full paths in naturally sorted order.
    """
    dir_path = Path(directory)
    files = [str(f) for f in dir_path.iterdir() if f.is_file()]
    return natsorted(files)


def encode_audio_to_base64(audio_path):
    """
    Read an audio file and encode it to base64 string.
    """
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    return audio_b64


def encode_audio_bytes_to_base64(audio_bytes):
    """
    Encode audio bytes to base64 string.
    
    Args:
        audio_bytes: Audio file content as bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(audio_bytes).decode("utf-8")


def transcribe_audio_streaming(runpod_client, audio_b64, task="Translate to English: ", temperature=0.1):
    """
    Stream audio transcription/translation using Sunflower Ultravox model.
    """
    stream = runpod_client.chat.completions.create(
        stream=True,
        model=os.getenv("MODEL_NAME", "jq/sunflower-ultravox-251111"),
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": [
                    { 
                        "type": "text",
                        "text": "You are Sunflower, a helpful assistant made by Sunbird AI who understands all Ugandan languages. You specialise in accurate translations, explanations, summaries and other language tasks.",
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text",
                        "text": task, 
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    }
                ]
            },
        ]
    )
    
    for event in stream:
        if event.choices[0].delta.content:
            yield event.choices[0].delta.content


@app.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """
    Query the vLLM server for available models
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vLLM models endpoint: {str(e)}"
        )

@app.get("/models/list")
async def list_model_names():
    """
    Get a simple list of model IDs/names
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{VLLM_BASE_URL}/v1/models")
            response.raise_for_status()
            data = response.json()
            model_names = [model["id"] for model in data.get("data", [])]
            return {"models": model_names, "count": len(model_names)}
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vLLM models endpoint: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Check if vLLM server is reachable
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_BASE_URL}/health")
            return {"status": "healthy", "vllm_status": response.status_code}
    except httpx.HTTPError as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Form(default="Translate to English: "),
    temperature: float = Form(default=0.1)
):
    """
    Transcribe or translate an audio file using Sunflower Ultravox model.
    
    Args:
        audio_file: Audio file upload (WAV format recommended)
        task: Task instruction. Options:
              - "Translate to English: " (default)
              - "Translate to [language]: "
              - "" for conversational response
        temperature: Sampling temperature (default: 0.1)
        
    Returns:
        Streaming text response
    """
    # Read the uploaded file
    audio_bytes = await audio_file.read()
    
    # Encode to base64
    audio_b64 = encode_audio_bytes_to_base64(audio_bytes)
    
    # Create a generator for streaming response
    def generate():
        for chunk in transcribe_audio_streaming(runpod_client, audio_b64, task, temperature):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Audio transcription API"}


if __name__ == "__main__":
    # audio_path = "audios/autism-child/autim-child-7.wav"
    # audio_b64 = encode_audio_to_base64(audio_path)

    # with open("audiobase64.txt", "wb") as f:
    #     f.write(audio_b64.encode("utf-8"))
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)