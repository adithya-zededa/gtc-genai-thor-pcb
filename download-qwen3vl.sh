#!/bin/bash

# Download Qwen3-VL model from Hugging Face
# This script downloads the model for local use or to prepare for deployment

set -e

# Configuration
REPO_ID="unsloth/Qwen3-VL-8B-Instruct-GGUF"
HF_TOKEN="${HF_TOKEN:-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}"
OUTPUT_DIR="${1:-./models/custom}"
QUANTIZATION="${2:-Q4_K_M}"

echo "=========================================="
echo "Qwen3-VL Model Downloader"
echo "=========================================="
echo "Repository: $REPO_ID"
echo "Output: $OUTPUT_DIR"
echo "Quantization: $QUANTIZATION"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed"
    exit 1
fi

# Check if huggingface-hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    pip3 install huggingface-hub
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download using Python script
python3 download_hf_model.py \
    --repo-id "$REPO_ID" \
    --hf-token "$HF_TOKEN" \
    --output-dir "$OUTPUT_DIR" \
    --quantization "$QUANTIZATION"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Download complete!"
    echo "=========================================="
    echo "Model files are in: $OUTPUT_DIR"
    echo ""
    echo "To use with the camera-agent:"
    echo "1. Build the app with custom model support:"
    echo "   docker build -t camera-agent-custom ."
    echo ""
    echo "2. Run with custom model:"
    echo "   docker run -v $(pwd)/models:/models \\"
    echo "     -e USE_CUSTOM_MODEL=true \\"
    echo "     -e VISION_MODEL=qwen3-vl-8b \\"
    echo "     -e CUSTOM_MODEL_DIR=/models/custom \\"
    echo "     camera-agent-custom"
else
    echo ""
    echo "✗ Download failed"
    exit 1
fi
