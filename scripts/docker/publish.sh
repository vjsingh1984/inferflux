#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DOCKER_REPO:-inferencial/inferflux}:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building Docker image ${IMAGE}"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/docker/Dockerfile" "${REPO_ROOT}"

if [ "${PUSH:-0}" = "1" ]; then
  echo "Pushing ${IMAGE}"
  docker push "${IMAGE}"
else
  echo "Skipping push (set PUSH=1 to enable)."
fi
