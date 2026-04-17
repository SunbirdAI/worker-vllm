output "service_url" {
  description = "Public URL of the Cloud Run service."
  value       = google_cloud_run_v2_service.app.uri
}

output "image" {
  description = "Full image reference deployed to Cloud Run."
  value       = local.image
}

output "repository" {
  description = "Artifact Registry repository path."
  value       = google_artifact_registry_repository.repo.id
}

output "region" {
  value = var.region
}

output "project_id" {
  value = var.project_id
}
