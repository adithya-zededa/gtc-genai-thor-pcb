#!/bin/bash

# Ollama Custom Model Deployment - Validation Script
# This script validates the deployment and checks all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RELEASE_NAME="${1:-my-ollama}"
NAMESPACE="${2:-default}"
COMPONENT="ollama-server"

echo -e "${BLUE}=== Ollama Deployment Validation ===${NC}"
echo -e "Release: ${RELEASE_NAME}"
echo -e "Namespace: ${NAMESPACE}"
echo ""

# Function to print status
print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
    else
        echo -e "${RED}✗${NC} $message"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if command_exists kubectl; then
    print_status "OK" "kubectl is installed"
else
    print_status "ERROR" "kubectl is not installed"
    exit 1
fi

if command_exists helm; then
    print_status "OK" "helm is installed"
else
    print_status "WARN" "helm is not installed (optional)"
fi

echo ""

# Get pod name
echo -e "${BLUE}Finding pod...${NC}"
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/component=$COMPONENT" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    print_status "ERROR" "No pod found with component=$COMPONENT"
    echo ""
    echo "Available pods:"
    kubectl get pods -n "$NAMESPACE"
    exit 1
fi

print_status "OK" "Found pod: $POD_NAME"
echo ""

# Check pod status
echo -e "${BLUE}Checking pod status...${NC}"
POD_STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}')
print_status "$([ "$POD_STATUS" = "Running" ] && echo OK || echo ERROR)" "Pod phase: $POD_STATUS"

POD_READY=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
print_status "$([ "$POD_READY" = "True" ] && echo OK || echo ERROR)" "Pod ready: $POD_READY"

echo ""

# Check init containers
echo -e "${BLUE}Checking init containers...${NC}"

INIT_CONTAINERS=("model-downloader" "model-copier" "modelfile-creator")
for container in "${INIT_CONTAINERS[@]}"; do
    STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath="{.status.initContainerStatuses[?(@.name=='$container')].state}" 2>/dev/null)
    if echo "$STATUS" | grep -q "terminated"; then
        REASON=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath="{.status.initContainerStatuses[?(@.name=='$container')].state.terminated.reason}" 2>/dev/null)
        if [ "$REASON" = "Completed" ]; then
            print_status "OK" "$container: Completed"
        else
            print_status "ERROR" "$container: $REASON"
        fi
    else
        print_status "WARN" "$container: $STATUS"
    fi
done

echo ""

# Check main containers
echo -e "${BLUE}Checking main containers...${NC}"

MAIN_CONTAINERS=("ollama-server" "model-loader")
for container in "${MAIN_CONTAINERS[@]}"; do
    STATUS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].ready}" 2>/dev/null)
    RESTARTS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath="{.status.containerStatuses[?(@.name=='$container')].restartCount}" 2>/dev/null)
    
    if [ "$STATUS" = "true" ]; then
        print_status "OK" "$container: Ready (restarts: $RESTARTS)"
    else
        print_status "ERROR" "$container: Not ready (restarts: $RESTARTS)"
    fi
done

echo ""

# Check volumes
echo -e "${BLUE}Checking volumes...${NC}"

VOLUMES=("ollama-data" "shared-storage")
for volume in "${VOLUMES[@]}"; do
    PVC=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath="{.spec.volumes[?(@.name=='$volume')].persistentVolumeClaim.claimName}" 2>/dev/null)
    if [ -n "$PVC" ]; then
        PVC_STATUS=$(kubectl get pvc "$PVC" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null)
        print_status "$([ "$PVC_STATUS" = "Bound" ] && echo OK || echo ERROR)" "$volume: $PVC ($PVC_STATUS)"
    else
        print_status "WARN" "$volume: Not using PVC"
    fi
done

echo ""

# Check GPU
echo -e "${BLUE}Checking GPU...${NC}"
if kubectl exec "$POD_NAME" -n "$NAMESPACE" -c ollama-server -- nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(kubectl exec "$POD_NAME" -n "$NAMESPACE" -c ollama-server -- nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
    print_status "OK" "GPU detected: $GPU_COUNT GPU(s) available"
else
    print_status "WARN" "GPU not detected or not accessible"
fi

echo ""

# Check Ollama server health
echo -e "${BLUE}Checking Ollama server health...${NC}"
if kubectl exec "$POD_NAME" -n "$NAMESPACE" -c ollama-server -- curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_status "OK" "Ollama API is responsive"
else
    print_status "ERROR" "Ollama API is not responding"
fi

echo ""

# List models
echo -e "${BLUE}Listing available models...${NC}"
MODELS=$(kubectl exec "$POD_NAME" -n "$NAMESPACE" -c ollama-server -- ollama list 2>/dev/null | tail -n +2)
if [ -n "$MODELS" ]; then
    print_status "OK" "Models found:"
    echo "$MODELS" | while IFS= read -r line; do
        echo "  $line"
    done
else
    print_status "WARN" "No models found"
fi

echo ""

# Check service
echo -e "${BLUE}Checking service...${NC}"
SERVICE_NAME=$(kubectl get service -n "$NAMESPACE" -l "app.kubernetes.io/component=$COMPONENT" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$SERVICE_NAME" ]; then
    SERVICE_TYPE=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.type}')
    SERVICE_PORT=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')
    print_status "OK" "Service: $SERVICE_NAME (type: $SERVICE_TYPE, port: $SERVICE_PORT)"
    
    if [ "$SERVICE_TYPE" = "NodePort" ]; then
        NODE_PORT=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].nodePort}')
        print_status "OK" "NodePort: $NODE_PORT"
    fi
else
    print_status "WARN" "Service not found"
fi

echo ""

# Check recent events
echo -e "${BLUE}Recent pod events:${NC}"
kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD_NAME" --sort-by='.lastTimestamp' | tail -5

echo ""

# Summary
echo -e "${BLUE}=== Validation Summary ===${NC}"

if [ "$POD_STATUS" = "Running" ] && [ "$POD_READY" = "True" ]; then
    echo -e "${GREEN}✓ Deployment appears to be healthy${NC}"
    echo ""
    echo "To test the model, run:"
    echo "  kubectl exec -it $POD_NAME -n $NAMESPACE -c ollama-server -- ollama run <model-name>"
    echo ""
    echo "To access the API locally:"
    echo "  kubectl port-forward $POD_NAME -n $NAMESPACE 11434:11434"
    echo "  curl http://localhost:11434/api/tags"
else
    echo -e "${RED}✗ Deployment has issues${NC}"
    echo ""
    echo "To debug, check logs:"
    echo "  kubectl logs $POD_NAME -n $NAMESPACE -c model-downloader"
    echo "  kubectl logs $POD_NAME -n $NAMESPACE -c ollama-server"
    echo "  kubectl logs $POD_NAME -n $NAMESPACE -c model-loader"
    echo ""
    echo "To describe the pod:"
    echo "  kubectl describe pod $POD_NAME -n $NAMESPACE"
fi

echo ""
