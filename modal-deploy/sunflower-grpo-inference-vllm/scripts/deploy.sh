#!/usr/bin/env bash
# Deploy the Sunflower 14B OpenAI-compatible vLLM app to Modal.
#
# Usage:
#   ./scripts/deploy.sh           # deploy the OpenAI app (default)
#   ./scripts/deploy.sh grpo      # deploy the GRPO LoRA app instead
#   ./scripts/deploy.sh both      # deploy both apps sequentially
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

OPENAI_APP="${ROOT}/sunflower_14b_openai_vllm_modal.py"
GRPO_APP="${ROOT}/sunflower_grpo_vllm_modal.py"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

deploy() {
  local target="$1"
  if [[ ! -f "${target}" ]]; then
    echo "error: app file not found: ${target}" >&2
    exit 1
  fi
  echo ">>> uv run modal deploy ${target}"
  uv run modal deploy "${target}"
}

case "${1:-openai}" in
  openai) deploy "${OPENAI_APP}" ;;
  grpo)   deploy "${GRPO_APP}" ;;
  both)   deploy "${OPENAI_APP}"; deploy "${GRPO_APP}" ;;
  *)      echo "usage: $0 [openai|grpo|both]" >&2; exit 2 ;;
esac
