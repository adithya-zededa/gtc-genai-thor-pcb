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

# The agent (text) model may live on a second pod. When AGENT_LLM_URL is
# unset both roles are served by VLLM_URL, and the second wait below is
# skipped rather than duplicating the first.
AGENT_URL=${AGENT_LLM_URL:-$VLLM_URL}

echo "vLLM Server (vision): $VLLM_URL"
echo "Vision Model: ${VISION_MODEL:-not set}"
if [ "$AGENT_URL" != "$VLLM_URL" ]; then
    echo "vLLM Server (agent):  $AGENT_URL"
    echo "Agent Model: ${AGENT_MODEL:-not set}"
else
    echo "Agent model: sharing the vision endpoint"
fi
echo "Timeout: ${VLLM_TIMEOUT:-300}s"
echo "Temperature: ${VLLM_TEMPERATURE:-0.1}"
echo ""

MAX_WAIT=${VLLM_WAIT_TIMEOUT:-600}  # 10 minutes default

# Wait for one vLLM endpoint to answer /health, up to MAX_WAIT.
# Non-fatal on timeout: the app degrades to "inference unavailable" and
# still serves the UI, logs, and history, which beats crash-looping while
# a model loads.
wait_for_vllm() {
    label=$1
    url=$2
    echo "Waiting for the $label vLLM server to be ready..."
    count=0
    while ! curl -s "$url/health" > /dev/null 2>&1; do
        if [ $count -ge "$MAX_WAIT" ]; then
            echo "Warning: $label vLLM not ready after ${MAX_WAIT}s, starting app anyway..."
            return 1
        fi
        echo "  Waiting for $label vLLM... (${count}s / ${MAX_WAIT}s)"
        sleep 10
        count=$((count + 10))
    done
    echo "✓ $label vLLM server is ready!"
    return 0
}

wait_for_vllm "vision" "$VLLM_URL" || true
if [ "$AGENT_URL" != "$VLLM_URL" ]; then
    wait_for_vllm "agent" "$AGENT_URL" || true
fi

echo ""
echo "Starting Web App..."
exec python run.py
