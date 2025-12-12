#!/bin/bash

# Build and push HF model downloader with versioned tags
set -e

VERSION="1.0.0"
REGISTRY="adithyazededa"
IMAGE_NAME="hf-model-downloader"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}"

echo "Building ${FULL_IMAGE}:${VERSION}"
docker build -f Dockerfile.hf-downloader -t "${FULL_IMAGE}:${VERSION}" .

echo "Tagging as latest"
docker tag "${FULL_IMAGE}:${VERSION}" "${FULL_IMAGE}:latest"

echo ""
echo "Built images:"
docker images | grep "${IMAGE_NAME}"
echo ""
echo "To push:"
echo "  docker push ${FULL_IMAGE}:${VERSION}"
echo "  docker push ${FULL_IMAGE}:latest"
