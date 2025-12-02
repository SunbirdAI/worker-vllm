# Cloud Run Deployment Plan for Sunflower Ultravox API

## Overview

Deploy the FastAPI-based audio transcription/translation service from `sunflower-ultravox-vllm/app.py` to Google Cloud Run. The application acts as a proxy to a RunPod-hosted vLLM inference server, handling audio file uploads and streaming transcription responses.

## Architecture

**Current State:**
- FastAPI app running locally on port 8001
- Connects to RunPod endpoint via OpenAI-compatible API
- Requires: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, MODEL_NAME

**Target State:**
- Containerized app on Cloud Run (auto-scaling)
- Secrets managed via Google Secret Manager
- HTTPS endpoint with managed certificates
- Production-ready with monitoring and logging

## Files to Create

**All files will be created within the `sunflower-ultravox-vllm/` folder to keep the deployment self-contained.**

### 1. Dockerfile
**Location:** `sunflower-ultravox-vllm/Dockerfile`

Multi-stage build optimized for Cloud Run:

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PATH=/root/.local/bin:$PATH
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Start with uvicorn (production server)
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info
```

**Key Features:**
- Multi-stage build reduces image size (~150MB vs ~900MB)
- Non-root user for security
- Cloud Run port 8080 (configurable via PORT env var)
- 2 Uvicorn workers for concurrency
- Health check for Cloud Run probes

### 2. .dockerignore
**Location:** `sunflower-ultravox-vllm/.dockerignore`

```
# Python
__pycache__/
*.py[cod]
env/
venv/

# CRITICAL: Exclude secrets
.env
.env.*

# Development files
test_app.py
*.ipynb
audios/
asserts/
audiobase64.txt
server.log
README.md
*.md

# Git
.git/
.gitignore
.DS_Store
```

**Security Note:** Prevents .env file with API keys from being baked into image.

### 3. cloudbuild.yaml (Optional - for CI/CD)
**Location:** `sunflower-ultravox-vllm/cloudbuild.yaml`

```yaml
steps:
  # Build container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:latest'
      - '.'
    dir: 'sunflower-ultravox-vllm'

  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:$COMMIT_SHA']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:latest']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'sunflower-ultravox-api'
      - '--image=gcr.io/$PROJECT_ID/sunflower-ultravox-api:$COMMIT_SHA'
      - '--region=europe-west1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--set-env-vars=MODEL_NAME=${_MODEL_NAME}'
      - '--set-secrets=RUNPOD_API_KEY=runpod-api-key:latest,RUNPOD_ENDPOINT_ID=runpod-endpoint-id:latest'
      - '--memory=1Gi'
      - '--cpu=1'
      - '--max-instances=10'
      - '--min-instances=0'
      - '--timeout=300'
      - '--port=8080'

substitutions:
  _MODEL_NAME: 'huwenjie333/sunflower32b-ultravox-251114-3'

images:
  - 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/sunflower-ultravox-api:latest'

timeout: 1800s
```

### 4. .gcloudignore
**Location:** `sunflower-ultravox-vllm/.gcloudignore`

```
.git/
.gitignore
__pycache__/
.env
env/
venv/
*.ipynb
audios/
asserts/
test_app.py
```

## Secret Management

### Setup Google Secret Manager

```bash
# Enable API
gcloud services enable secretmanager.googleapis.com

# Create secrets (extract from .env file)
echo -n "YOUR_RUNPOD_API_KEY" | gcloud secrets create runpod-api-key \
  --data-file=- \
  --replication-policy="automatic"

echo -n "YOUR_ENDPOINT_ID" | gcloud secrets create runpod-endpoint-id \
  --data-file=- \
  --replication-policy="automatic"

# Grant Cloud Run access
PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding runpod-api-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding runpod-endpoint-id \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

**Secret Strategy:**
- `RUNPOD_API_KEY` → Secret Manager (sensitive)
- `RUNPOD_ENDPOINT_ID` → Secret Manager (semi-sensitive)
- `MODEL_NAME` → Environment variable (non-sensitive)

## Cloud Run Configuration

### Resource Allocation

```
CPU: 1 vCPU
Memory: 1 GiB
Min Instances: 0 (scale to zero)
Max Instances: 10
Concurrency: 80 requests/instance
Timeout: 300s (5 minutes)
Port: 8080
Region: europe-west1 (Belgium)
```

**Rationale:**
- Start with minimal resources (1 vCPU, 1 GiB) for cost efficiency
- Audio files can be several MB (base64-encoded)
- Streaming responses need persistent connections
- 5-minute timeout handles long audio files
- Scale to zero reduces costs during idle periods
- europe-west1 provides standard pricing with EU data residency

### Cost Estimate

**Example (100 requests/day, 2s avg processing, 1 vCPU, 1 GiB):**
- Monthly cost: ~$1.50-3 (likely within free tier)
- Free tier: 2M requests/month, 360k GiB-seconds/month, 180k vCPU-seconds/month

**Optimization:**
- CPU throttling enabled (90% idle cost reduction)
- Scale to zero when no traffic
- Right-sized resources (can start smaller and scale up)

## Deployment Steps

### Prerequisites

```bash
# Install gcloud CLI (if needed)
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project and region
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region europe-west1

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

### Option 1: Direct Deployment (Recommended for First Deploy)

```bash
cd /Users/patrickwalukagga/Projects/Sunbirdai/worker-vllm/sunflower-ultravox-vllm

gcloud run deploy sunflower-ultravox-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_NAME=huwenjie333/sunflower32b-ultravox-251114-3 \
  --set-secrets RUNPOD_API_KEY=runpod-api-key:latest,RUNPOD_ENDPOINT_ID=runpod-endpoint-id:latest \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 300 \
  --port 8080
```

**What happens:**
1. Cloud Build automatically detects and builds Dockerfile
2. Pushes image to Google Container Registry
3. Deploys to Cloud Run
4. Returns HTTPS service URL

### Option 2: Manual Build + Deploy (More Control)

```bash
# Build image
gcloud builds submit \
  --tag gcr.io/YOUR_PROJECT_ID/sunflower-ultravox-api:v1.0.0 \
  /Users/patrickwalukagga/Projects/Sunbirdai/worker-vllm/sunflower-ultravox-vllm

# Deploy image
gcloud run deploy sunflower-ultravox-api \
  --image gcr.io/YOUR_PROJECT_ID/sunflower-ultravox-api:v1.0.0 \
  --region europe-west1 \
  --platform managed \
  --set-env-vars MODEL_NAME=huwenjie333/sunflower32b-ultravox-251114-3 \
  --set-secrets RUNPOD_API_KEY=runpod-api-key:latest,RUNPOD_ENDPOINT_ID=runpod-endpoint-id:latest \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

### Option 3: CI/CD with Cloud Build Triggers

```bash
# Connect GitHub repo and create trigger
gcloud beta builds triggers create github \
  --repo-name=worker-vllm \
  --repo-owner=SunbirdAI \
  --branch-pattern="^main$" \
  --build-config=sunflower-ultravox-vllm/cloudbuild.yaml

# Future deployments: just push to main branch
git push origin main
```

## Post-Deployment Verification

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe sunflower-ultravox-api \
  --region europe-west1 \
  --format 'value(status.url)')

echo "Service URL: $SERVICE_URL"

# Test endpoints
curl $SERVICE_URL/health
curl $SERVICE_URL/
curl $SERVICE_URL/models/list

# Test transcription
curl -X POST $SERVICE_URL/transcribe \
  -F "audio_file=@/path/to/audio.wav" \
  -F "task=Translate to English: " \
  -F "temperature=0.1"

# View logs
gcloud run services logs read sunflower-ultravox-api \
  --region europe-west1 \
  --limit 50
```

## Security Considerations

### API Access Control

**Current Plan:** Public API (`--allow-unauthenticated`)

**Alternatives:**
1. **Authenticated only:** Remove `--allow-unauthenticated` (requires IAM token)
2. **API Key authentication:** Add middleware in app.py
3. **Internal only:** `--ingress internal` (VPC-restricted)

### File Upload Security

**Add to app.py:**
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}

# Validate in /transcribe endpoint
if len(audio_bytes) > MAX_FILE_SIZE:
    raise HTTPException(status_code=413, detail="File too large")
```

### Secret Rotation

```bash
# Update secret (creates new version)
echo -n "NEW_API_KEY" | gcloud secrets versions add runpod-api-key --data-file=-

# Cloud Run automatically uses latest version
```

## Monitoring and Logging

### Cloud Logging

Application logs automatically sent to Cloud Logging. View in Console or CLI:

```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=sunflower-ultravox-api AND resource.labels.location=europe-west1" \
  --limit 50
```

### Key Metrics to Monitor

- Request count and latency (p50, p95, p99)
- Error rate (5xx responses)
- CPU and memory utilization
- Cold start frequency
- RunPod API success rate

### Budget Alerts

```bash
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Sunflower API Budget" \
  --budget-amount=50 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

## Deployment Checklist

### Pre-Deployment
- [ ] Update requirements.txt (add `uvicorn[standard]`)
- [ ] Create Dockerfile in `sunflower-ultravox-vllm/`
- [ ] Create .dockerignore in `sunflower-ultravox-vllm/`
- [ ] Create .gcloudignore in `sunflower-ultravox-vllm/`
- [ ] Create cloudbuild.yaml in `sunflower-ultravox-vllm/` (optional, for CI/CD)
- [ ] Verify .env is in .gitignore (already done)
- [ ] Enable GCP APIs (run, cloudbuild, secretmanager)
- [ ] Create secrets in Secret Manager
- [ ] Grant IAM permissions

### Deployment
- [ ] Build and test container locally (optional)
- [ ] Deploy to Cloud Run (europe-west1, 1 vCPU, 1 GiB)
- [ ] Test all endpoints (/health, /, /transcribe, /models)
- [ ] Verify secrets loaded correctly
- [ ] Check logs for errors

### Post-Deployment
- [ ] Configure custom domain (sunflower-ultravox.sunbird.ai)
- [ ] Update DNS records (CNAME to ghs.googlehosted.com)
- [ ] Wait for SSL certificate provisioning
- [ ] Set up monitoring dashboard
- [ ] Configure budget alerts
- [ ] Document service URL
- [ ] (Optional) Configure CI/CD pipeline with Cloud Build triggers

## Code Modifications

**All code changes will be made within the `sunflower-ultravox-vllm/` folder.**

### Required: Update requirements.txt

Add `uvicorn[standard]` to the requirements.txt file for production server features:

**Current:**
```
fastapi
openai
python-dotenv
natsort
runpod
httpx
```

**Updated:**
```
fastapi
uvicorn[standard]
openai
python-dotenv
natsort
runpod
httpx
```

**Rationale:** The Dockerfile CMD uses uvicorn directly, so it must be in requirements.txt. The `[standard]` extras include performance optimizations (uvloop, httptools).

### Optional: Make Port Configurable in app.py

**Current (line 218):**
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Recommended:**
```python
import os
port = int(os.getenv("PORT", 8001))
uvicorn.run(app, host="0.0.0.0", port=port)
```

**Note:** This is optional since the Dockerfile CMD uses uvicorn directly, but makes local development more flexible.

### Optional: Add File Size Validation

Add to `/transcribe` endpoint for production hardening:

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

@app.post("/transcribe")
async def transcribe_audio(...):
    audio_bytes = await audio_file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    # ... rest of function
```

## Troubleshooting

### Port Binding Error
**Error:** Container failed to start
**Solution:** Ensure Dockerfile CMD uses `--host 0.0.0.0` (not localhost)

### Secret Not Found
**Error:** Secret "runpod-api-key" not found
**Solution:** Verify secret exists and service account has secretAccessor role

### Memory Exceeded
**Error:** Container exceeded memory limit
**Solution:** Increase memory allocation or optimize file handling

### RunPod Connection Error
**Error:** Failed to connect to RunPod
**Solution:** Verify RUNPOD_ENDPOINT_ID and API key are correct in Secret Manager

## Quick Start Commands

```bash
# 1. Navigate to project
cd /Users/patrickwalukagga/Projects/Sunbirdai/worker-vllm/sunflower-ultravox-vllm

# 2. Set GCP project and region
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region europe-west1

# 3. Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# 4. Create secrets (use actual values from your .env file)
echo -n "YOUR_RUNPOD_API_KEY" | gcloud secrets create runpod-api-key --data-file=-
echo -n "YOUR_RUNPOD_ENDPOINT_ID" | gcloud secrets create runpod-endpoint-id --data-file=-

# 5. Deploy to Cloud Run
gcloud run deploy sunflower-ultravox-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_NAME=huwenjie333/sunflower32b-ultravox-251114-3 \
  --set-secrets RUNPOD_API_KEY=runpod-api-key:latest,RUNPOD_ENDPOINT_ID=runpod-endpoint-id:latest \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300

# 6. Get service URL
gcloud run services describe sunflower-ultravox-api --region europe-west1 --format 'value(status.url)'

# 7. Configure custom domain
gcloud run domain-mappings create \
  --service sunflower-ultravox-api \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1

# 8. Get DNS configuration
gcloud run domain-mappings describe \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1
```

**Note:** After step 8, update your DNS provider with the CNAME record pointing to `ghs.googlehosted.com`.

## Cost Optimization Tips

1. **Start small:** Begin with 1 vCPU, 1 GiB and scale up if needed
2. **Scale to zero:** Min instances = 0 (no charges when idle)
3. **CPU throttling:** Enabled by default (90% idle cost reduction)
4. **Monitor usage:** Set up budget alerts to avoid surprises
5. **Region selection:** europe-west1 provides standard pricing with EU data residency

## Custom Domain Setup

Map `sunflower-ultravox.sunbird.ai` to the Cloud Run service:

```bash
# Step 1: Create domain mapping
gcloud run domain-mappings create \
  --service sunflower-ultravox-api \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1

# Step 2: Get DNS records to configure
gcloud run domain-mappings describe \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1
```

**DNS Configuration:**
The command above will output DNS records. Add these to your DNS provider (sunbird.ai):
- Type: CNAME
- Name: sunflower-ultravox
- Value: ghs.googlehosted.com (or specific value from output)

**Verification:**
Cloud Run automatically provisions and manages SSL certificates for custom domains. HTTPS will be enabled automatically once DNS propagates (typically 5-15 minutes).

## Next Steps After Deployment

1. **Custom domain:** Configure DNS for sunflower-ultravox.sunbird.ai (see above)
2. **API authentication:** Add API key middleware for production security
3. **Rate limiting:** Implement Cloud Armor or app-level rate limiting
4. **Caching:** Add cache headers for /models endpoint
5. **CI/CD:** Set up automated deployments from GitHub
6. **Monitoring:** Set up alerts for errors and high latency
