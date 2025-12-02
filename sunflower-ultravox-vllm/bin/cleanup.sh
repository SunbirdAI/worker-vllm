#!/bin/bash

# Sunflower Ultravox API - Cloud Run Cleanup Script
# This script helps delete deployed resources from Google Cloud Run

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
CUSTOM_DOMAIN="sunflower-ultravox.sunbird.ai"
REPO="sunflower-repo"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
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

# Get GCP project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    print_error "No GCP project is set. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_info "Using GCP project: $PROJECT_ID"

# Function to check if service exists
check_service_exists() {
    if gcloud run services describe "$SERVICE_NAME" --region "$REGION" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to delete Cloud Run service
delete_service() {
    print_warning "This will delete the Cloud Run service: $SERVICE_NAME"
    echo ""

    if ! check_service_exists; then
        print_warning "Service '$SERVICE_NAME' does not exist in region $REGION"
        return 0
    fi

    # Get service URL before deletion
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --format 'value(status.url)' 2>/dev/null)

    if [ -n "$SERVICE_URL" ]; then
        print_info "Current service URL: $SERVICE_URL"
    fi

    echo ""
    read -p "Are you sure you want to delete this service? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_info "Deletion cancelled"
        return 0
    fi

    print_info "Deleting Cloud Run service..."
    gcloud run services delete "$SERVICE_NAME" \
        --region "$REGION" \
        --quiet

    print_success "Cloud Run service deleted successfully"
}

# Function to delete domain mapping
delete_domain_mapping() {
    print_info "Checking for domain mapping..."

    if ! gcloud run domain-mappings describe --domain "$CUSTOM_DOMAIN" --region "$REGION" >/dev/null 2>&1; then
        print_warning "Domain mapping '$CUSTOM_DOMAIN' does not exist"
        return 0
    fi

    print_warning "This will delete the domain mapping: $CUSTOM_DOMAIN"
    echo ""
    read -p "Do you want to delete the domain mapping? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_info "Domain deletion cancelled"
        return 0
    fi

    print_info "Deleting domain mapping..."
    gcloud run domain-mappings delete \
        --domain "$CUSTOM_DOMAIN" \
        --region "$REGION" \
        --quiet

    print_success "Domain mapping deleted successfully"
    echo ""
    print_info "Remember to remove the CNAME record from your DNS provider (sunbird.ai)"
}

# Function to delete container images
delete_images() {
    print_info "Checking for container images in Artifact Registry..."

    # Check if repository exists
    if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
        print_warning "Artifact Registry repository '$REPO' does not exist"
        return 0
    fi

    # List images
    IMAGES=$(gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}" --format="value(package)" 2>/dev/null | head -1)

    if [ -z "$IMAGES" ]; then
        print_warning "No container images found for $SERVICE_NAME in Artifact Registry"

        # Also check old GCR images
        print_info "Checking for old GCR images..."
        OLD_IMAGES=$(gcloud container images list --repository=gcr.io/$PROJECT_ID --filter="name:sunflower-ultravox-api" --format="value(name)" 2>/dev/null)

        if [ -z "$OLD_IMAGES" ]; then
            print_info "No old GCR images found either"
            return 0
        else
            print_info "Found old GCR images"
            echo ""
            read -p "Do you want to delete old GCR images? (yes/no): " confirm
            if [ "$confirm" = "yes" ]; then
                gcloud container images delete gcr.io/$PROJECT_ID/sunflower-ultravox-api --quiet 2>/dev/null || true
                print_success "Old GCR images deleted"
            fi
        fi
        return 0
    fi

    echo ""
    print_info "Found the following images in Artifact Registry:"
    gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}" --format="table(version,createTime,updateTime)" 2>/dev/null || true

    echo ""
    print_warning "This will delete ALL container images for $SERVICE_NAME"
    read -p "Do you want to delete all container images? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        print_info "Image deletion cancelled"
        return 0
    fi

    print_info "Deleting container images from Artifact Registry..."
    gcloud artifacts docker images delete "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}" --delete-tags --quiet 2>/dev/null || true

    print_success "Container images deleted successfully"

    # Ask if they want to delete the repository too
    echo ""
    read -p "Do you want to delete the entire Artifact Registry repository? (yes/no): " confirm_repo
    if [ "$confirm_repo" = "yes" ]; then
        print_info "Deleting Artifact Registry repository..."
        gcloud artifacts repositories delete "$REPO" --location="$REGION" --quiet
        print_success "Repository deleted"
    fi
}

# Function to list build history
list_builds() {
    print_info "Fetching recent Cloud Build history..."

    gcloud builds list \
        --filter="images:gcr.io/$PROJECT_ID/sunflower-ultravox-api" \
        --limit=10 \
        --format="table(id,status,createTime,duration)" 2>/dev/null || {
        print_warning "No build history found or Cloud Build API not enabled"
    }
}

# Function to show current status
show_status() {
    echo ""
    echo "========================================="
    echo "  Current Deployment Status"
    echo "========================================="

    # Check service
    if check_service_exists; then
        SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
            --region "$REGION" \
            --format 'value(status.url)' 2>/dev/null)
        echo "✓ Service: $SERVICE_NAME (DEPLOYED)"
        echo "  Region: $REGION"
        echo "  URL: $SERVICE_URL"
    else
        echo "✗ Service: $SERVICE_NAME (NOT DEPLOYED)"
    fi

    echo ""

    # Check domain mapping
    if gcloud run domain-mappings describe --domain "$CUSTOM_DOMAIN" --region "$REGION" >/dev/null 2>&1; then
        echo "✓ Domain: $CUSTOM_DOMAIN (MAPPED)"
    else
        echo "✗ Domain: $CUSTOM_DOMAIN (NOT MAPPED)"
    fi

    echo ""

    # Check images in Artifact Registry
    if gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
        IMAGE_COUNT=$(gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}" --format="value(version)" 2>/dev/null | wc -l)
        if [ "$IMAGE_COUNT" -gt 0 ]; then
            echo "✓ Container Images: $IMAGE_COUNT image(s) in Artifact Registry"
        else
            echo "✗ Container Images: None in Artifact Registry"
        fi
    else
        echo "✗ Artifact Registry: Repository not found"
        # Check old GCR
        OLD_IMAGE_COUNT=$(gcloud container images list-tags gcr.io/$PROJECT_ID/sunflower-ultravox-api --format="value(digest)" 2>/dev/null | wc -l)
        if [ "$OLD_IMAGE_COUNT" -gt 0 ]; then
            echo "⚠ Old GCR Images: $OLD_IMAGE_COUNT image(s)"
        fi
    fi

    echo "========================================="
}

# Function to delete everything
delete_all() {
    echo ""
    print_warning "========================================="
    print_warning "  COMPLETE CLEANUP"
    print_warning "========================================="
    print_warning "This will delete:"
    echo "  1. Cloud Run service: $SERVICE_NAME"
    echo "  2. Domain mapping: $CUSTOM_DOMAIN"
    echo "  3. All container images"
    echo ""
    print_error "This action cannot be undone!"
    echo ""
    read -p "Type 'DELETE' to confirm complete cleanup: " confirm

    if [ "$confirm" != "DELETE" ]; then
        print_info "Cleanup cancelled"
        return 0
    fi

    echo ""
    print_info "Starting complete cleanup..."
    echo ""

    # Delete in reverse order of creation
    delete_domain_mapping
    echo ""
    delete_service
    echo ""
    delete_images

    echo ""
    print_success "Complete cleanup finished!"
    echo ""
    print_info "Don't forget to:"
    echo "  - Remove DNS records from your provider"
    echo "  - Clean up any Cloud Build triggers if configured"
}

# Main menu
show_menu() {
    echo ""
    echo "========================================="
    echo "  Sunflower Ultravox API Cleanup"
    echo "========================================="
    echo "1) Show current status"
    echo "2) Delete Cloud Run service only"
    echo "3) Delete domain mapping only"
    echo "4) Delete container images only"
    echo "5) List build history"
    echo "6) Delete everything (service + domain + images)"
    echo "7) Exit"
    echo "========================================="
}

# Main script execution
main() {
    # If arguments provided, run non-interactive mode
    if [ $# -gt 0 ]; then
        case "$1" in
            status)
                show_status
                ;;
            service)
                delete_service
                ;;
            domain)
                delete_domain_mapping
                ;;
            images)
                delete_images
                ;;
            builds)
                list_builds
                ;;
            all)
                delete_all
                ;;
            *)
                print_error "Unknown command: $1"
                echo "Usage: $0 {status|service|domain|images|builds|all}"
                echo ""
                echo "Commands:"
                echo "  status  - Show deployment status"
                echo "  service - Delete Cloud Run service"
                echo "  domain  - Delete domain mapping"
                echo "  images  - Delete container images"
                echo "  builds  - List build history"
                echo "  all     - Delete everything"
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
                show_status
                ;;
            2)
                delete_service
                ;;
            3)
                delete_domain_mapping
                ;;
            4)
                delete_images
                ;;
            5)
                list_builds
                ;;
            6)
                delete_all
                ;;
            7)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please select 1-7."
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
