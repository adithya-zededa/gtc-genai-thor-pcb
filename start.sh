#!/bin/bash
set -e

echo "=== ZEDEDA Camera Monitoring Agent ==="
echo ""

# Create data directories if they don't exist
DATA_DIR=${CAMERA_AGENT_DATA_DIR:-/app/data}
mkdir -p "$DATA_DIR"
mkdir -p "${DETECTED_IMAGES_DIR:-$DATA_DIR/detected_images}"
mkdir -p "${PROCESSED_FRAMES_DIR:-$DATA_DIR/processed_frames}"

# Verify vLLM configuration
if [ -z "$VLLM_URL" ]; then
    echo "Error: VLLM_URL environment variable is required"
    echo "This image requires an external vLLM server for inference"
    exit 1
fi

echo "vLLM Server: $VLLM_URL"
echo "Vision Model: ${VISION_MODEL:-not set}"
echo "Timeout: ${VLLM_TIMEOUT:-300}s"
echo "Temperature: ${VLLM_TEMPERATURE:-0.1}"
echo ""

# Wait for vLLM server to be available (with timeout)
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=${VLLM_WAIT_TIMEOUT:-600}  # 10 minutes default
WAIT_COUNT=0
while ! curl -s "$VLLM_URL/health" > /dev/null 2>&1; do
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "Warning: vLLM server not ready after ${MAX_WAIT}s, starting app anyway..."
        break
    fi
    echo "  Waiting for vLLM... (${WAIT_COUNT}s / ${MAX_WAIT}s)"
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 10))
done

if curl -s "$VLLM_URL/health" > /dev/null 2>&1; then
    echo "✓ vLLM server is ready!"
fi

echo ""
echo "Starting Web App..."
exec python run.py
