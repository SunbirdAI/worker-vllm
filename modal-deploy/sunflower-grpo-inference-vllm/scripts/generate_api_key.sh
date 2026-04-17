#!/usr/bin/env bash
# Generate a 32-character URL-safe API key for the vLLM OpenAI server.
#
# Usage:
#   ./scripts/generate_api_key.sh              # print the key
#   ./scripts/generate_api_key.sh --create     # create the Modal secret `vllm-api-key`
#   ./scripts/generate_api_key.sh --update     # overwrite the existing Modal secret
#   ./scripts/generate_api_key.sh --env        # append VLLM_API_KEY=... to .env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

SECRET_NAME="vllm-api-key"

# 24 random bytes -> 32 base64url chars (no padding).
KEY="$(openssl rand -base64 24 | tr '+/' '-_' | tr -d '=\n')"

case "${1:-}" in
  --create)
    uv run modal secret create "${SECRET_NAME}" "VLLM_API_KEY=${KEY}"
    echo
    echo "Created Modal secret '${SECRET_NAME}'. Save this key — it will not be shown again:"
    echo "  VLLM_API_KEY=${KEY}"
    ;;
  --update)
    uv run modal secret create --force "${SECRET_NAME}" "VLLM_API_KEY=${KEY}"
    echo
    echo "Updated Modal secret '${SECRET_NAME}'. Save this key — it will not be shown again:"
    echo "  VLLM_API_KEY=${KEY}"
    ;;
  --env)
    touch "${ENV_FILE}"
    # Drop any existing VLLM_API_KEY line, then append the new one.
    if grep -q '^VLLM_API_KEY=' "${ENV_FILE}"; then
      tmp="$(mktemp)"
      grep -v '^VLLM_API_KEY=' "${ENV_FILE}" > "${tmp}"
      mv "${tmp}" "${ENV_FILE}"
    fi
    echo "VLLM_API_KEY=${KEY}" >> "${ENV_FILE}"
    echo "Wrote VLLM_API_KEY to ${ENV_FILE}"
    echo "  VLLM_API_KEY=${KEY}"
    ;;
  "")
    echo "${KEY}"
    ;;
  *)
    echo "unknown flag: $1" >&2
    echo "usage: $0 [--create|--update|--env]" >&2
    exit 2
    ;;
esac
