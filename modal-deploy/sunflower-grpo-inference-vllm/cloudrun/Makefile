# Build/push/deploy helpers for the Cloud Run proxy.
#
# Usage:
#   make auth          # gcloud auth + Artifact Registry docker auth
#   make build         # build linux/amd64 image locally
#   make push          # push image to Artifact Registry
#   make tf-init       # terraform init
#   make tf-plan       # terraform plan (requires terraform.tfvars)
#   make tf-apply      # terraform apply
#   make deploy        # build + push + tf-apply
#   make url           # print the service URL
#   make logs          # tail recent Cloud Run logs

APP        ?= sunflower-grpo-test
PROJECT_ID ?= sb-gcp-project-01
REGION     ?= europe-west1
PORT       ?= 8080
REPO       ?= sunflower-grpo-test

# Tagging strategy:
#   - clean working tree → <short-sha>
#   - dirty working tree → <short-sha>-<UTC timestamp>   (so Terraform sees a real change)
#   - no git             → <UTC timestamp>
GIT_SHA    := $(shell git rev-parse --short HEAD 2>/dev/null)
GIT_DIRTY  := $(shell git status --porcelain 2>/dev/null)
TS         := $(shell date -u +%Y%m%d%H%M%S)
ifeq ($(GIT_SHA),)
  AUTO_TAG := $(TS)
else ifeq ($(GIT_DIRTY),)
  AUTO_TAG := $(GIT_SHA)
else
  AUTO_TAG := $(GIT_SHA)-$(TS)
endif
TAG        ?= $(AUTO_TAG)

IMAGE      := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(APP)
IMAGE_TAG  := $(IMAGE):$(TAG)
IMAGE_LATEST := $(IMAGE):latest

# fastapi_app.py + Dockerfile live one level up from this Makefile.
BUILD_CONTEXT := ..

.DEFAULT_GOAL := help

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: auth
auth: ## gcloud auth + configure docker for Artifact Registry
	gcloud auth login
	gcloud config set project $(PROJECT_ID)
	gcloud auth configure-docker $(REGION)-docker.pkg.dev --quiet

.PHONY: build
build: ## Build linux/amd64 image
	docker buildx build \
		--platform linux/amd64 \
		-t $(IMAGE_TAG) \
		-t $(IMAGE_LATEST) \
		--load \
		$(BUILD_CONTEXT)

.PHONY: push
push: ## Push both SHA and latest tags
	docker push $(IMAGE_TAG)
	docker push $(IMAGE_LATEST)

.PHONY: build-push
build-push: ## Build + push in one buildx invocation (no local load)
	docker buildx build \
		--platform linux/amd64 \
		-t $(IMAGE_TAG) \
		-t $(IMAGE_LATEST) \
		--push \
		$(BUILD_CONTEXT)

.PHONY: tf-init
tf-init: ## terraform init
	terraform init

.PHONY: tf-plan
tf-plan: ## terraform plan
	terraform plan -var="image_tag=$(TAG)"

.PHONY: tf-apply
tf-apply: ## terraform apply
	terraform apply -auto-approve -var="image_tag=$(TAG)"

.PHONY: tf-destroy
tf-destroy: ## terraform destroy
	terraform destroy -var="image_tag=$(TAG)"

.PHONY: deploy
deploy: build-push tf-apply ## Build, push, and apply in one shot

.PHONY: url
url: ## Print the Cloud Run service URL
	@terraform output -raw service_url

.PHONY: logs
logs: ## Tail recent Cloud Run logs
	gcloud run services logs read $(APP) \
		--region $(REGION) \
		--project $(PROJECT_ID) \
		--limit 100
