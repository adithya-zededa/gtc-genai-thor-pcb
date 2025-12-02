#!/bin/bash
set -e

# Create data directories if they don't exist
DATA_DIR=${CAMERA_AGENT_DATA_DIR:-/app/data}
mkdir -p "$DATA_DIR"
mkdir -p "${DETECTED_IMAGES_DIR:-$DATA_DIR/detected_images}"
mkdir -p "${PROCESSED_FRAMES_DIR:-$DATA_DIR/processed_frames}"

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready."

# Pull required models
# This ensures the models are available before the app starts.
# If the volume is persisted, this will be fast on subsequent runs.
echo "Checking/Pulling models..."

# Use environment variables if set, otherwise use defaults
VISION_MODEL=${VISION_MODEL:-qwen3-vl:8b}

echo "Pulling vision model: $VISION_MODEL..."
ollama pull "$VISION_MODEL"

# Start the application
echo "Starting Web App..."
# Ensure we use the installed python3
exec python3 web_app.py
