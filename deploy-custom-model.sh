#!/bin/bash

# Deploy camera-agent with Qwen3-VL custom model
# This script helps you set up the custom model workflow

set -e

echo "================================================"
echo "Camera Agent - Qwen3-VL Custom Model Setup"
echo "================================================"
echo ""

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo "✓ $message"
    elif [ "$status" = "INFO" ]; then
        echo "→ $message"
    else
        echo "✗ $message"
    fi
}

# Check prerequisites
print_status "INFO" "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    print_status "ERROR" "python3 is required"
    exit 1
fi
print_status "OK" "python3 found"

if ! command -v docker &> /dev/null; then
    print_status "ERROR" "docker is required"
    exit 1
fi
print_status "OK" "docker found"

echo ""

# Step 1: Download model from Hugging Face
print_status "INFO" "Step 1: Download Qwen3-VL model from Hugging Face"
echo ""

if [ -d "./models/custom" ] && [ "$(ls -A ./models/custom/*.gguf 2>/dev/null)" ]; then
    print_status "OK" "Model already downloaded in ./models/custom"
    ls -lh ./models/custom/*.gguf
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "INFO" "Skipping download"
    else
        print_status "INFO" "Downloading model..."
        ./download-qwen3vl.sh ./models/custom Q4_K_M
    fi
else
    print_status "INFO" "Downloading model (this may take 10-20 minutes)..."
    ./download-qwen3vl.sh ./models/custom Q4_K_M
fi

echo ""

# Step 2: Build Docker image
print_status "INFO" "Step 2: Build Docker image"
echo ""

read -p "Build Docker image? (Y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_status "INFO" "Skipping build"
else
    print_status "INFO" "Building camera-agent image..."
    docker build -t camera-agent-qwen3vl:latest .
    print_status "OK" "Image built successfully"
fi

echo ""

# Step 3: Deploy
print_status "INFO" "Step 3: Choose deployment method"
echo ""
echo "1) Docker Compose (recommended for local)"
echo "2) Kubernetes/Helm"
echo "3) Exit"
echo ""
read -p "Select option (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        print_status "INFO" "Using Docker Compose..."
        echo ""
        
        # Update docker-compose to use custom model
        if grep -q "USE_CUSTOM_MODEL=false" docker-compose.yml; then
            print_status "INFO" "Enabling custom model in docker-compose.yml..."
            sed -i 's/USE_CUSTOM_MODEL=false/USE_CUSTOM_MODEL=true/' docker-compose.yml
        fi
        
        print_status "INFO" "Starting services..."
        docker-compose up -d
        
        echo ""
        print_status "OK" "Deployment started!"
        echo ""
        echo "Logs: docker-compose logs -f"
        echo "Web UI: http://localhost:8080"
        echo ""
        ;;
    2)
        print_status "INFO" "Using Kubernetes/Helm..."
        echo ""
        
        # Build and push HF downloader image
        print_status "INFO" "Building HF model downloader image..."
        ./build-hf-downloader.sh
        
        read -p "Push image to registry? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker push adithyazededa/hf-model-downloader:latest
        fi
        
        echo ""
        print_status "INFO" "Deploy with Helm:"
        echo ""
        echo "  helm install camera-agent ./helm/camera-agent \\"
        echo "    -f ./helm/camera-agent/values-qwen3vl-hf.yaml"
        echo ""
        echo "Or to use standalone Ollama server:"
        echo ""
        echo "  helm install ollama-server ./helm/camera-agent \\"
        echo "    -f ./helm/camera-agent/values-qwen3vl-hf.yaml \\"
        echo "    --set inferenceBackend=ollama"
        echo ""
        ;;
    3)
        print_status "INFO" "Exiting..."
        exit 0
        ;;
    *)
        print_status "ERROR" "Invalid option"
        exit 1
        ;;
esac

echo ""
print_status "OK" "Setup complete!"
echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Monitor logs to ensure model loads:"
echo "   docker-compose logs -f camera-agent"
echo ""
echo "2. Check model is loaded:"
echo "   docker exec camera-agent ollama list"
echo ""
echo "3. Access the web interface:"
echo "   http://localhost:8080"
echo ""
echo "4. The model 'qwen3-vl:8b' will be used for vision tasks"
echo "   (Change via VISION_MODEL environment variable)"
echo ""
