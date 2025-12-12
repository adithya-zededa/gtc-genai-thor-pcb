#!/bin/bash
set -e

# Create data directories if they don't exist
DATA_DIR=${CAMERA_AGENT_DATA_DIR:-/app/data}
mkdir -p "$DATA_DIR"
mkdir -p "${DETECTED_IMAGES_DIR:-$DATA_DIR/detected_images}"
mkdir -p "${PROCESSED_FRAMES_DIR:-$DATA_DIR/processed_frames}"

# Set Ollama context length (default 32000)
export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-32000}
echo "Ollama context length: $OLLAMA_CONTEXT_LENGTH"

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done
echo "Ollama is ready."

# Model loading logic
VISION_MODEL=${VISION_MODEL}
CUSTOM_MODEL_DIR=${CUSTOM_MODEL_DIR:-/models/custom}
USE_CUSTOM_MODEL=${USE_CUSTOM_MODEL:-false}

if [ -z "$VISION_MODEL" ]; then
    echo "Error: VISION_MODEL environment variable is required"
    exit 1
fi

echo "Loading vision model: $VISION_MODEL..."

# Check if custom model directory exists and has model files
if [ "$USE_CUSTOM_MODEL" = "true" ] && [ -d "$CUSTOM_MODEL_DIR" ]; then
    echo "Custom model mode enabled. Checking for model files in $CUSTOM_MODEL_DIR..."
    
    # Find model file (GGUF, bin, safetensors, etc.)
    MODEL_FILE=$(find "$CUSTOM_MODEL_DIR" -maxdepth 2 -type f \( -name "*.gguf" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.pt" -o -name "*.pth" \) | head -1)
    
    if [ -n "$MODEL_FILE" ]; then
        echo "Found custom model file: $MODEL_FILE"
        
        # Check if Modelfile exists, create if not
        MODELFILE="$CUSTOM_MODEL_DIR/Modelfile"
        if [ ! -f "$MODELFILE" ]; then
            echo "Creating Modelfile for custom model..."
            cat > "$MODELFILE" << EOF
FROM $MODEL_FILE

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_ctx ${OLLAMA_CONTEXT_LENGTH}
PARAMETER repeat_penalty 1.05

SYSTEM """You are Qwen3-VL, a vision-language AI assistant. You can analyze images and provide detailed descriptions, answer questions about visual content, and help with various vision-related tasks."""
EOF
            echo "Modelfile created at $MODELFILE"
        else
            echo "Using existing Modelfile: $MODELFILE"
        fi
        
        # Check if model already exists in Ollama
        if ollama list | grep -q "^$VISION_MODEL"; then
            echo "Model $VISION_MODEL already exists in Ollama"
        else
            echo "Creating custom model $VISION_MODEL from Modelfile..."
            ollama create "$VISION_MODEL" -f "$MODELFILE"
            
            if [ $? -eq 0 ]; then
                echo "✓ Custom model $VISION_MODEL created successfully"
            else
                echo "✗ Failed to create custom model, falling back to pull"
                ollama pull "$VISION_MODEL"
            fi
        fi
    else
        echo "No model files found in $CUSTOM_MODEL_DIR, falling back to pull"
        ollama pull "$VISION_MODEL"
    fi
else
    # Standard mode: pull from Ollama registry
    echo "Standard mode: Pulling vision model from Ollama registry..."
    ollama pull "$VISION_MODEL"
fi

# Verify model is loaded
echo "Verifying model availability..."
ollama list

# Start the application
echo "Starting Web App..."
# Ensure we use the installed python3
exec python3 web_app.py
