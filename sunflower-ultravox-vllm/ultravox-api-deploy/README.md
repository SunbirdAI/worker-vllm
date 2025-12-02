# Sunflower Ultravox Audio Transcription API

A production-ready deployment for serving the Sunflower Ultravox model with vLLM and a FastAPI wrapper for audio transcription and translation.

## Overview

This project deploys two services:

- **vLLM Server**: GPU-accelerated inference server running the Sunflower Ultravox multimodal model
- **FastAPI Application**: REST API wrapper providing audio transcription and translation endpoints

## Prerequisites

- Ubuntu 20.04+ (or similar Linux distribution)
- NVIDIA GPU with CUDA support
- Docker and Docker Compose v2
- NVIDIA Container Toolkit
- At least 40GB+ GPU VRAM (for 32B model)

### Installing NVIDIA Container Toolkit

```bash
# Add the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Project Structure

```
.
├── docker-compose.yml    # Service orchestration
├── Dockerfile            # FastAPI application image
├── Dockerfile.vllm       # vLLM with audio support
├── app.py                # FastAPI application code
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
└── README.md
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# .env
HF_TOKEN=your_huggingface_token_here
```

### Customizable Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace API token for model access |
| `VLLM_BASE_URL` | `http://vllm:8000` | Internal vLLM server URL |
| `MODEL_NAME` | `huwenjie333/sunflower32b-ultravox-251114-3` | Model identifier |

### vLLM Server Options

Modify the `command` section in `docker-compose.yml` to adjust:

```yaml
command: >
  --model huwenjie333/sunflower32b-ultravox-251114-3
  --port 8000
  --trust-remote-code
  --max-model-len 4096          # Maximum context length
  --tensor-parallel-size 1      # Number of GPUs for tensor parallelism
  --gpu-memory-utilization 0.9  # GPU memory fraction to use
```

## Deployment

### Quick Start

```bash
# Clone or copy files to your server
cd /path/to/project

# Create environment file
echo "HF_TOKEN=your_token_here" > .env

# Build and start services
docker compose up -d

# View logs
docker compose logs -f
```

### Build Only

```bash
# Build images without starting
docker compose build

# Build specific service
docker compose build vllm
docker compose build api
```

### Managing Services

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Restart a specific service
docker compose restart api

# View status
docker compose ps

# View logs for specific service
docker compose logs -f vllm
docker compose logs -f api
```

## API Reference

### Health Endpoints

#### Root Health Check

```
GET /
```

Response:
```json
{"status": "ok", "message": "Audio transcription API"}
```

#### Detailed Health Check

```
GET /health
```

Response:
```json
{"status": "healthy", "vllm_status": 200}
```

### Model Information

#### List Models (Full Details)

```
GET /models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "huwenjie333/sunflower32b-ultravox-251114-3",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

#### List Model Names

```
GET /models/list
```

Response:
```json
{
  "models": ["huwenjie333/sunflower32b-ultravox-251114-3"],
  "count": 1
}
```

### Transcription

#### Transcribe/Translate Audio

```
POST /transcribe
Content-Type: multipart/form-data
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio_file` | file | Yes | - | Audio file (WAV recommended) |
| `task` | string | No | `"Translate to English: "` | Task instruction |
| `temperature` | float | No | `0.1` | Sampling temperature |

**Task Options:**

- `"Translate to English: "` - Translate audio to English
- `"Translate to Luganda: "` - Translate audio to Luganda
- `"Transcribe: "` - Transcribe in original language
- `""` (empty) - Conversational response about the audio

### Example Usage

**cURL:**

```bash
curl -X POST "http://localhost:8001/transcribe" \
  -F "audio_file=@audio.wav" \
  -F "task=Translate to English: " \
  -F "temperature=0.1"
```

**Python:**

```python
import requests

url = "http://localhost:8001/transcribe"
files = {"audio_file": open("audio.wav", "rb")}
data = {"task": "Translate to English: ", "temperature": 0.1}

response = requests.post(url, files=files, data=data, stream=True)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="", flush=True)
```

**JavaScript:**

```javascript
const formData = new FormData();
formData.append('audio_file', audioFile);
formData.append('task', 'Translate to English: ');

const response = await fetch('http://localhost:8001/transcribe', {
  method: 'POST',
  body: formData
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  console.log(decoder.decode(value));
}
```

## Ports

| Service | Internal Port | External Port | Description |
|---------|---------------|---------------|-------------|
| vLLM | 8000 | 8000 | OpenAI-compatible API |
| FastAPI | 8001 | 8001 | Transcription API |

## Troubleshooting

### Common Issues

#### Permission Denied (Docker Socket)

```
permission denied while trying to connect to the Docker daemon socket
```

**Fix:**

```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```

#### NVIDIA Runtime Not Found

```
unknown or invalid runtime name: nvidia
```

**Fix:** Install NVIDIA Container Toolkit (see Prerequisites section)

#### Out of Memory (OOM)

```
CUDA out of memory
```

**Fix:** Reduce model memory usage in `docker-compose.yml`:

```yaml
command: >
  --model huwenjie333/sunflower32b-ultravox-251114-3
  --max-model-len 2048
  --gpu-memory-utilization 0.85
```

#### vLLM Audio Support Missing

```
Please install vllm[audio] for audio support
```

**Fix:** Ensure you're using the custom `Dockerfile.vllm` that installs audio support.

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f vllm
docker compose logs -f api

# Last 100 lines
docker compose logs --tail 100 vllm
```

### Checking GPU Usage

```bash
# On host
nvidia-smi

# Inside container
docker compose exec vllm nvidia-smi
```

## Production Considerations

### Security

1. **Restrict CORS**: Update `app.py` to limit allowed origins:

   ```python
   allow_origins=["https://yourdomain.com"]
   ```

2. **Add API Authentication**: Enable vLLM API key:

   ```yaml
   command: >
     --model huwenjie333/sunflower32b-ultravox-251114-3
     --api-key ${VLLM_API_KEY}
   ```

3. **Use Nginx/Traefik**: Add TLS termination and rate limiting

### Monitoring

Add Prometheus metrics export:

```yaml
command: >
  --model huwenjie333/sunflower32b-ultravox-251114-3
  --enable-metrics
```

### Scaling

For multi-GPU setups:

```yaml
command: >
  --model huwenjie333/sunflower32b-ultravox-251114-3
  --tensor-parallel-size 2  # Use 2 GPUs
```

## Files Reference

### docker-compose.yml

```yaml
services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command: >
      --model huwenjie333/sunflower32b-ultravox-251114-3
      --port 8000
      --trust-remote-code
      --max-model-len 4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - VLLM_BASE_URL=http://vllm:8000
      - MODEL_NAME=huwenjie333/sunflower32b-ultravox-251114-3
    depends_on:
      vllm:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Dockerfile.vllm

```dockerfile
FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir "vllm[audio]"
```

### requirements.txt

```
fastapi
uvicorn[standard]
python-dotenv
openai
httpx
natsort
python-multipart
```

## Support

For issues related to:

- **Sunflower model**: Contact Sunbird AI
- **vLLM**: https://github.com/vllm-project/vllm
- **This deployment**: pwalukagga@sunbird.ai