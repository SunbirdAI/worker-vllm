#!/bin/bash

# Sunflower Ultravox API - Cloud Run Deployment Script
# This script automates the deployment of the FastAPI application to Google Cloud Run

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="sunflower-ultravox-api"
REGION="europe-west1"
MODEL_NAME="huwenjie333/sunflower32b-ultravox-251114-3"
REPO="sunflower-repo"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists gcloud; then
    print_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_success "gcloud CLI is installed"

# Check if .env file exists in sunflower-ultravox-vllm directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    print_warning ".env file not found at $PROJECT_DIR/.env. You'll need to provide environment variables manually."
fi

# Get GCP project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    print_error "No GCP project is set. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_info "Using GCP project: $PROJECT_ID"

# Function to setup GCP project
setup_gcp_project() {
    print_info "Setting up GCP project configuration..."

    gcloud config set run/region "$REGION"
    print_success "Region set to $REGION"

    print_info "Enabling required APIs (this may take a few minutes)..."
    gcloud services enable run.googleapis.com \
        cloudbuild.googleapis.com \
        artifactregistry.googleapis.com

    print_success "APIs enabled successfully"

    # Create Artifact Registry repository if it doesn't exist
    print_info "Setting up Artifact Registry repository..."
    if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
        print_info "Creating Artifact Registry repository: $REPO"
        gcloud artifacts repositories create "$REPO" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Docker repository for Sunflower Ultravox API"
        print_success "Artifact Registry repository created"
    else
        print_info "Artifact Registry repository already exists"
    fi
}

# Function to read environment variables from .env file
read_env_vars() {
    print_info "Reading environment variables from .env file..."

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        print_error ".env file not found at $PROJECT_DIR/.env. Please create it with RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"
        return 1
    fi

    # Extract values from .env file
    export RUNPOD_API_KEY=$(grep RUNPOD_API_KEY "$PROJECT_DIR/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)
    export RUNPOD_ENDPOINT_ID=$(grep RUNPOD_ENDPOINT_ID "$PROJECT_DIR/.env" | cut -d '=' -f2 | tr -d '"' | tr -d "'" | xargs)

    if [ -z "$RUNPOD_API_KEY" ] || [ -z "$RUNPOD_ENDPOINT_ID" ]; then
        print_error "RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not found in .env file at $PROJECT_DIR/.env"
        return 1
    fi

    print_success "Environment variables loaded successfully"
    print_info "RUNPOD_API_KEY: ${RUNPOD_API_KEY:0:10}... (truncated)"
    print_info "RUNPOD_ENDPOINT_ID: $RUNPOD_ENDPOINT_ID"
}

# Function to build and push Docker image
build_image() {
    # Define image tag
    local IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}:latest"

    print_info "Building Docker image..."
    print_info "Image will be tagged as: $IMAGE_TAG"
    print_info "Building from: $PROJECT_DIR"

    # Build and push using Cloud Build (redirect all output to stderr)
    cd "$PROJECT_DIR"
    if gcloud builds submit --tag "$IMAGE_TAG" . >&2; then
        print_success "Docker image built and pushed successfully"
        # Return only the image tag to stdout
        echo "$IMAGE_TAG"
        return 0
    else
        print_error "Failed to build Docker image"
        return 1
    fi
}

# Function to deploy to Cloud Run
deploy_to_cloudrun() {
    print_info "Deploying to Cloud Run..."

    # Read environment variables if not already loaded
    if [ -z "$RUNPOD_API_KEY" ] || [ -z "$RUNPOD_ENDPOINT_ID" ]; then
        read_env_vars || return 1
    fi

    # Build and push image first
    print_info "Step 1: Building and pushing Docker image..."

    # Capture the image tag (only stdout, stderr goes to terminal)
    IMAGE_TAG=$(build_image)
    BUILD_STATUS=$?

    if [ $BUILD_STATUS -ne 0 ]; then
        print_error "Failed to build image"
        return 1
    fi

    print_info "Using image: $IMAGE_TAG"

    # Deploy the image
    print_info "Step 2: Deploying to Cloud Run in $REGION..."
    gcloud run deploy "$SERVICE_NAME" \
        --image "$IMAGE_TAG" \
        --platform managed \
        --region "$REGION" \
        --allow-unauthenticated \
        --set-env-vars "MODEL_NAME=$MODEL_NAME,RUNPOD_API_KEY=$RUNPOD_API_KEY,RUNPOD_ENDPOINT_ID=$RUNPOD_ENDPOINT_ID" \
        --memory 1Gi \
        --cpu 1 \
        --max-instances 10 \
        --min-instances 0 \
        --timeout 300 \
        --port 8080

    print_success "Deployment completed successfully!"
}

# Function to get service URL
get_service_url() {
    print_info "Retrieving service URL..."

    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format 'value(status.url)' 2>/dev/null)

    if [ -n "$SERVICE_URL" ]; then
        print_success "Service is available at: $SERVICE_URL"
        echo ""
        print_info "You can test the endpoints:"
        echo "  Health check: curl $SERVICE_URL/health"
        echo "  API root:     curl $SERVICE_URL/"
        echo "  Models list:  curl $SERVICE_URL/models/list"
    else
        print_error "Could not retrieve service URL"
    fi
}

# Function to setup custom domain
setup_custom_domain() {
    print_info "Setting up custom domain: sunflower-ultravox.sunbird.ai"

    # Create domain mapping (using beta for --region support)
    if gcloud beta run domain-mappings describe --domain sunflower-ultravox.sunbird.ai --region "$REGION" >/dev/null 2>&1; then
        print_warning "Domain mapping already exists"
    else
        gcloud beta run domain-mappings create \
            --service "$SERVICE_NAME" \
            --domain sunflower-ultravox.sunbird.ai \
            --region "$REGION"
    fi

    # Get DNS configuration
    print_info "DNS configuration:"
    gcloud beta run domain-mappings describe \
        --domain sunflower-ultravox.sunbird.ai \
        --region "$REGION"

    echo ""
    print_info "Add the following DNS record to your DNS provider (sunbird.ai):"
    echo "  Type: CNAME"
    echo "  Name: sunflower-ultravox"
    echo "  Value: ghs.googlehosted.com"
    echo ""
    print_info "SSL certificate will be automatically provisioned once DNS propagates (5-15 minutes)"
}

# Function to view logs
view_logs() {
    print_info "Viewing recent logs..."
    gcloud run services logs read "$SERVICE_NAME" \
        --region "$REGION" \
        --limit 50
}

# Main menu
show_menu() {
    echo ""
    echo "========================================="
    echo "  Sunflower Ultravox API Deployment"
    echo "========================================="
    echo "1) Full setup (first-time deployment)"
    echo "2) Deploy/Update service only"
    echo "3) Setup custom domain"
    echo "4) View service URL"
    echo "5) View logs"
    echo "6) Exit"
    echo "========================================="
}

# Main script execution
main() {
    # If arguments provided, run non-interactive mode
    if [ $# -gt 0 ]; then
        case "$1" in
            setup)
                setup_gcp_project
                deploy_to_cloudrun
                get_service_url
                ;;
            deploy)
                deploy_to_cloudrun
                get_service_url
                ;;
            domain)
                setup_custom_domain
                ;;
            url)
                get_service_url
                ;;
            logs)
                view_logs
                ;;
            *)
                print_error "Unknown command: $1"
                echo "Usage: $0 {setup|deploy|domain|url|logs}"
                exit 1
                ;;
        esac
        exit 0
    fi

    # Interactive mode
    while true; do
        show_menu
        read -p "Select an option [1-7]: " choice

        case $choice in
            1)
                setup_gcp_project
                deploy_to_cloudrun
                get_service_url
                echo ""
                read -p "Do you want to setup custom domain now? (y/n): " setup_domain
                if [ "$setup_domain" = "y" ] || [ "$setup_domain" = "Y" ]; then
                    setup_custom_domain
                fi
                ;;
            2)
                deploy_to_cloudrun
                get_service_url
                ;;
            3)
                setup_custom_domain
                ;;
            4)
                get_service_url
                ;;
            5)
                view_logs
                ;;
            6)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-6."
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
