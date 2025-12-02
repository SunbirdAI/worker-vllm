# Cloud Run Deployment Guide

This guide explains how to deploy the Sunflower Ultravox API to Google Cloud Run using **environment variables** (no Secret Manager required).

## Prerequisites

1. **Google Cloud CLI** installed and authenticated
   ```bash
   gcloud auth login
   ```

2. **GCP Project** set up
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **.env file** in the `sunflower-ultravox-vllm` directory with:
   ```
   RUNPOD_API_KEY=your_api_key_here
   RUNPOD_ENDPOINT_ID=your_endpoint_id_here
   MODEL_NAME=huwenjie333/sunflower32b-ultravox-251114-3
   ```

## Quick Deploy

### Using the Deployment Script (Recommended)

```bash
cd sunflower-ultravox-vllm/bin
./deploy.sh setup
```

This will:
- ✅ Enable required GCP APIs
- ✅ Read environment variables from `.env`
- ✅ Build and deploy the Docker container
- ✅ Display the service URL

### Manual Deployment

If you prefer to run commands manually:

```bash
cd sunflower-ultravox-vllm

# 1. Set region
gcloud config set run/region europe-west1

# 2. Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 3. Load environment variables
source .env

# 4. Deploy
gcloud run deploy sunflower-ultravox-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars "MODEL_NAME=$MODEL_NAME,RUNPOD_API_KEY=$RUNPOD_API_KEY,RUNPOD_ENDPOINT_ID=$RUNPOD_ENDPOINT_ID" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

## Deployment Script Commands

The `bin/deploy.sh` script supports both interactive and command-line modes:

### Interactive Mode
```bash
cd sunflower-ultravox-vllm/bin
./deploy.sh
```

### Command-Line Mode
```bash
cd sunflower-ultravox-vllm/bin

# First-time setup
./deploy.sh setup

# Update existing deployment
./deploy.sh deploy

# Setup custom domain
./deploy.sh domain

# View service URL
./deploy.sh url

# View logs
./deploy.sh logs
```

## Configuration

### Resource Settings
- **Region**: europe-west1 (Belgium)
- **CPU**: 1 vCPU
- **Memory**: 1 GiB
- **Min Instances**: 0 (scale to zero)
- **Max Instances**: 10
- **Timeout**: 300 seconds (5 minutes)

### Environment Variables
All configuration is done via environment variables:
- `MODEL_NAME` - The model identifier
- `RUNPOD_API_KEY` - Your RunPod API key
- `RUNPOD_ENDPOINT_ID` - Your RunPod endpoint ID

**Note**: Environment variables are passed directly to Cloud Run. No Secret Manager is used.

## Custom Domain Setup

### Step 1: Verify Domain Ownership

Before mapping a custom domain, you need to verify ownership of `sunbird.ai`:

```bash
# Open the domain verification page
gcloud domains verify sunbird.ai
```

This will give you a TXT record to add to your DNS provider. Follow the instructions to verify the domain.

Alternatively, verify via the Google Search Console:
1. Go to https://search.google.com/search-console
2. Add and verify `sunbird.ai`
3. Once verified, it will be available for Cloud Run domain mapping

**Note**: If you've already verified `sunbird.ai` for other services (like `lamwo.sunbird.ai` or `suntrace.sunbird.ai`), you may need to re-verify or ensure the verification is active for your current GCP project.

### Step 2: Create Domain Mapping

Once the domain is verified:

```bash
# Create domain mapping (using beta for region support)
gcloud beta run domain-mappings create \
  --service sunflower-ultravox-api \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1

# Get DNS configuration
gcloud beta run domain-mappings describe \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1
```

### Step 3: Configure DNS

Add a CNAME record to your DNS provider (sunbird.ai):
- **Type**: CNAME
- **Name**: sunflower-ultravox
- **Value**: ghs.googlehosted.com

### Step 4: Wait for SSL Certificate

SSL certificates are automatically provisioned by Cloud Run once DNS propagates (typically 5-15 minutes).

You can check the status with:
```bash
gcloud beta run domain-mappings describe \
  --domain sunflower-ultravox.sunbird.ai \
  --region europe-west1
```

## Testing the Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe sunflower-ultravox-api \
  --region europe-west1 \
  --format 'value(status.url)')

# Health check
curl $SERVICE_URL/health

# Root endpoint
curl $SERVICE_URL/

# List models
curl $SERVICE_URL/models/list

# Test transcription
curl -X POST $SERVICE_URL/transcribe \
  -F "audio_file=@audios/context_eng_7.wav" \
  -F "task=Translate to English: " \
  -F "temperature=0.1"
```

## Viewing Logs

```bash
# Recent logs
gcloud run services logs read sunflower-ultravox-api \
  --region europe-west1 \
  --limit 50

# Real-time logs
gcloud run services logs tail sunflower-ultravox-api \
  --region europe-west1
```

## Updating the Service

To deploy code changes:

```bash
cd sunflower-ultravox-vllm/bin
./deploy.sh deploy
```

Or manually:

```bash
cd sunflower-ultravox-vllm
source .env

gcloud run deploy sunflower-ultravox-api \
  --source . \
  --region europe-west1 \
  --set-env-vars "MODEL_NAME=$MODEL_NAME,RUNPOD_API_KEY=$RUNPOD_API_KEY,RUNPOD_ENDPOINT_ID=$RUNPOD_ENDPOINT_ID"
```

## CI/CD with Cloud Build

For automated deployments from Git pushes, use `cloudbuild.yaml`:

```bash
# Create trigger
gcloud beta builds triggers create github \
  --repo-name=worker-vllm \
  --repo-owner=SunbirdAI \
  --branch-pattern="^main$" \
  --build-config=sunflower-ultravox-vllm/cloudbuild.yaml \
  --substitutions _RUNPOD_API_KEY="YOUR_API_KEY",_RUNPOD_ENDPOINT_ID="YOUR_ENDPOINT_ID"
```

**Important**: Set the substitution variables in your Cloud Build trigger settings to keep credentials secure.

## Troubleshooting

### Port Binding Error
**Error**: Container failed to start
**Solution**: Ensure Dockerfile uses `--host 0.0.0.0` (already configured)

### Environment Variables Not Set
**Error**: Missing RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID
**Solution**: Verify `.env` file exists and contains the required values

### Build Timeout
**Error**: Build exceeded timeout
**Solution**: Increase timeout in cloudbuild.yaml or use manual build

### Memory Exceeded
**Error**: Container exceeded memory limit
**Solution**: Increase memory allocation to 2Gi:
```bash
gcloud run services update sunflower-ultravox-api \
  --region europe-west1 \
  --memory 2Gi
```

## Cost Optimization

With 1 vCPU and 1 GiB:
- **Estimated cost**: $1.50-3/month for moderate usage
- **Free tier**: 2M requests/month, 360k GiB-seconds/month
- **Scale to zero**: No charges when idle
- **CPU throttling**: 90% cost reduction during idle periods

## Security Notes

- Environment variables are visible in the Cloud Run console
- For production, consider using Secret Manager (requires IAM setup)
- `.env` file is excluded from Git via `.gitignore`
- `.dockerignore` prevents `.env` from being baked into images

## Files Overview

- `Dockerfile` - Multi-stage container build
- `.dockerignore` - Excludes dev files and secrets from image
- `.gcloudignore` - Excludes files from gcloud deployments
- `cloudbuild.yaml` - CI/CD pipeline configuration
- `bin/deploy.sh` - Automated deployment script
- `DEPLOYMENT.md` - This file

## Support

For detailed planning documentation, see:
`/Users/patrickwalukagga/.claude/plans/vast-finding-lagoon.md`
