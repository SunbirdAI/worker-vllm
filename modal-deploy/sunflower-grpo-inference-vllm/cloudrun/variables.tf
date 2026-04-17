variable "project_id" {
  type        = string
  description = "GCP project ID."
  default     = "sb-gcp-project-01"
}

variable "region" {
  type        = string
  description = "GCP region for Cloud Run and Artifact Registry."
  default     = "europe-west1"
}

variable "app" {
  type        = string
  description = "Cloud Run service name / image name."
  default     = "sunflower-grpo-test"
}

variable "repo" {
  type        = string
  description = "Artifact Registry repository name."
  default     = "sunflower-grpo-test"
}

variable "image_tag" {
  type        = string
  description = "Image tag to deploy (e.g. the git SHA or `latest`)."
  default     = "latest"
}

variable "port" {
  type        = number
  description = "Container port the app listens on."
  default     = 8080
}

variable "sunflower_upstream_url" {
  type        = string
  description = "Upstream Modal web URL used by the proxy."
  default     = "https://sb-modal-ws--sunflower-grpo-vllm-web.modal.run"
}

variable "sunbird_prod_url" {
  type        = string
  description = "Production Sunbird AI API base URL."
  default     = "https://api.sunbird.ai"
}

variable "sunbird_prod_token" {
  type        = string
  description = "Bearer token for the production Sunbird AI API. Passed as an env var on the Cloud Run service."
  sensitive   = true
}

variable "upstream_timeout" {
  type        = number
  description = "Per-request upstream timeout in seconds."
  default     = 600
}

variable "rate_limit_per_minute" {
  type        = string
  description = "slowapi per-IP per-minute limit."
  default     = "100/minute"
}

variable "rate_limit_per_day" {
  type        = string
  description = "slowapi per-IP per-day limit."
  default     = "1000/day"
}

variable "cpu" {
  type        = string
  description = "Cloud Run CPU allocation per container."
  default     = "1"
}

variable "memory" {
  type        = string
  description = "Cloud Run memory allocation per container."
  default     = "1Gi"
}

variable "min_instances" {
  type        = number
  description = "Minimum number of Cloud Run instances."
  default     = 0
}

variable "max_instances" {
  type        = number
  description = "Maximum number of Cloud Run instances."
  default     = 5
}

variable "request_timeout_seconds" {
  type        = number
  description = "Cloud Run request timeout in seconds."
  default     = 600
}

variable "allow_public" {
  type        = bool
  description = "If true, binds `allUsers` as run.invoker (public service)."
  default     = true
}

variable "deletion_protection" {
  type        = bool
  description = "Cloud Run v2 deletion_protection flag. Defaults to false for this test service; set true for production to prevent Terraform from destroying/recreating it."
  default     = false
}
