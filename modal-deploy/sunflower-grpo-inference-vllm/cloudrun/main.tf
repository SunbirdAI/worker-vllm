provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repo}/${var.app}:${var.image_tag}"
}

# ---------------------------------------------------------------------------
# Enable required APIs
# ---------------------------------------------------------------------------
resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Artifact Registry repository
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.repo
  description   = "Container images for ${var.app}"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

# ---------------------------------------------------------------------------
# Cloud Run v2 service
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "app" {
  name                = var.app
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = var.deletion_protection

  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    timeout = "${var.request_timeout_seconds}s"

    containers {
      image = local.image

      ports {
        container_port = var.port
      }

      resources {
        limits = {
          cpu    = var.cpu
          memory = var.memory
        }
      }

      # NOTE: PORT is a reserved Cloud Run env var and is automatically set by
      # the runtime to match `ports.container_port` — setting it here is
      # rejected by the API.
      env {
        name  = "SUNFLOWER_UPSTREAM_URL"
        value = var.sunflower_upstream_url
      }
      env {
        name  = "SUNBIRD_PROD_URL"
        value = var.sunbird_prod_url
      }
      env {
        name  = "SUNBIRD_PROD_TOKEN"
        value = var.sunbird_prod_token
      }
      env {
        name  = "UPSTREAM_TIMEOUT"
        value = tostring(var.upstream_timeout)
      }
      env {
        name  = "RATE_LIMIT_PER_MINUTE"
        value = var.rate_limit_per_minute
      }
      env {
        name  = "RATE_LIMIT_PER_DAY"
        value = var.rate_limit_per_day
      }

      startup_probe {
        initial_delay_seconds = 5
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 10
        tcp_socket {
          port = var.port
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_project_service.run,
    google_artifact_registry_repository.repo,
  ]
}

# ---------------------------------------------------------------------------
# Public access (allUsers → run.invoker)
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_public ? 1 : 0
  location = google_cloud_run_v2_service.app.location
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
