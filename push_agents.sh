#!/usr/bin/env bash
set -euo pipefail

# Push all Radar miner agent images to GitHub Container Registry (ghcr.io)
# and make them public.
#
# Prerequisites:
#   - docker logged in to ghcr.io:
#       echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
#   - GITHUB_TOKEN needs: write:packages, delete:packages scopes
#
# Usage:
#   GITHUB_ORG=tensorlink-ai ./push_agents.sh

GITHUB_ORG="${GITHUB_ORG:?Set GITHUB_ORG (e.g. tensorlink-ai)}"
REGISTRY="ghcr.io/${GITHUB_ORG}"
TAG="${TAG:-latest}"
BUILD_DIR="$(cd "$(dirname "$0")/radar-miners" && pwd)"

AGENTS=(frontier_sniper bucket_specialist pareto_hunter)

echo "==> Building and pushing agents to ${REGISTRY}"

for agent in "${AGENTS[@]}"; do
    image="${REGISTRY}/radar-miner-${agent}:${TAG}"
    echo ""
    echo "--- ${agent} ---"

    echo "[build] ${image}"
    docker build -f "${BUILD_DIR}/agents/${agent}/Dockerfile" \
        -t "${image}" \
        "${BUILD_DIR}"

    echo "[push]  ${image}"
    docker push "${image}"

    echo "[public] Setting visibility to public"
    # Uses GitHub API to flip the package to public.
    # Requires a token with write:packages scope in GITHUB_TOKEN env var.
    curl -sf -X PUT \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/orgs/${GITHUB_ORG}/packages/container/radar-miner-${agent}/visibility" \
        -d '{"visibility":"public"}' \
        && echo " -> public" \
        || echo " -> WARN: could not set public (check GITHUB_TOKEN scopes)"
done

echo ""
echo "==> Done. Images:"
for agent in "${AGENTS[@]}"; do
    echo "  ${REGISTRY}/radar-miner-${agent}:${TAG}"
done
