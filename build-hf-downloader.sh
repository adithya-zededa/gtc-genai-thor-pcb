#!/bin/bash

# Build and push Hugging Face model downloader image
# Usage: ./build-hf-downloader.sh [registry/repository] [tag]

set -e

REGISTRY="${1:-adithyazededa/hf-model-downloader}"
TAG="${2:-1.0.0}"
IMAGE="${REGISTRY}:${TAG}"

echo "Building Hugging Face model downloader image: ${IMAGE}"
echo ""

# Build the image
docker build -f Dockerfile.hf-downloader -t "${IMAGE}" .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful: ${IMAGE}"
    echo ""
    echo "To push the image, run:"
    echo "  docker push ${IMAGE}"
    echo ""
    echo "To test locally:"
    echo "  docker run --rm ${IMAGE} --help"
    echo ""
    echo "To test downloading:"
    echo "  docker run --rm -v \$(pwd)/models:/models ${IMAGE} \\"
    echo "    --repo-id unsloth/Qwen3-VL-8B-Instruct-GGUF \\"
    echo "    --hf-token YOUR_TOKEN \\"
    echo "    --output-dir /models \\"
    echo "    --quantization Q4_K_M"
else
    echo "✗ Build failed"
    exit 1
fi
