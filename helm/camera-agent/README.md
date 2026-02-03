# ZEDEDA Camera Monitoring Agent - Helm Chart

AI-powered camera monitoring agent with agentic VLM capabilities for real-time shipping box detection and label verification, designed for NVIDIA Jetson devices.

## Overview

This Helm chart deploys the **ZEDEDA Camera Monitoring Agent**, a system that captures video from cameras, detects packaging/shipping boxes using Vision Language Models (VLMs), and raises alerts when boxes are missing visible shipping labels.

### Components Deployed

| Component | Description |
|-----------|-------------|
| **Camera Agent** | Flask + SocketIO web application for monitoring, configuration, and real-time analysis |
| **vLLM Server** | GPU-accelerated vision language model server using NVIDIA Triton (Qwen3-VL-8B-Instruct) |

### Key Features

- **Smart Frame Preprocessing**: SSIM-based scene change detection reduces VLM calls by ~95%
- **Real-time Video Feed**: Live camera stream with WebSocket updates
- **Agentic Tool Calling**: LLM-driven automated actions (email alerts, evidence saving)
- **Confidence Blending**: Combines VLM confidence with classical CV analyzers
- **Alert Routing**: Automatic email/desktop notifications for unlabeled boxes
- **Web Dashboard**: Modern UI for monitoring, logs, settings, and user management

## Prerequisites

- Kubernetes cluster (K3s recommended for Jetson)
- NVIDIA GPU with driver and container toolkit
- Helm 3.x
- Camera device (USB or CSI)

### Jetson Thor Specific Requirements

```bash
# Verify GPU is accessible
nvidia-smi

# Check for camera device
ls -la /dev/video*

# Ensure NVIDIA container runtime is configured
docker info | grep -i nvidia
```

## Quick Start

### Install with default settings

```bash
# Install from local directory
helm install camera-agent ./camera-agent

# Or install with custom values
helm install camera-agent ./camera-agent \
  --set image.tag=v4 \
  --set vllmServer.args.maxModelLen=32768
```

### Access the application

```bash
# Get the NodePort
export NODE_PORT=$(kubectl get svc camera-agent -o jsonpath='{.spec.ports[0].nodePort}')
export NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[0].address}')
echo "Open http://$NODE_IP:$NODE_PORT"
```

## Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Camera agent image | `adithyazededa/camera-agent-thor-vllm` |
| `image.tag` | Image version | `v4` |
| `camera.enabled` | Mount camera device | `true` |
| `camera.devicePath` | Camera device path | `/dev/video0` |
| `camera.index` | OpenCV camera index | `0` |
| `vllmServer.enabled` | Deploy vLLM server | `true` |
| `vllmServer.model` | VLM model to serve | `Qwen/Qwen3-VL-8B-Instruct` |
| `vllmServer.timeout` | Inference timeout (seconds) | `300` |
| `vllmServer.temperature` | Generation temperature | `0.1` |
| `vllmServer.args.maxModelLen` | Max context length | `4096` |
| `vllmServer.args.gpuMemoryUtilization` | GPU memory usage (0.0-1.0) | `0.85` |
| `service.type` | Service type | `NodePort` |
| `service.nodePort` | External port | `30080` |

### vLLM Server Configuration

The vLLM server uses NVIDIA Triton Server with vLLM backend:

```yaml
vllmServer:
  enabled: true
  model: "Qwen/Qwen3-VL-8B-Instruct"
  timeout: 300
  temperature: 0.1
  
  image:
    repository: nvcr.io/nvidia/tritonserver
    tag: "25.12-vllm-python-py3"
  
  args:
    tensorParallelSize: 1
    gpuMemoryUtilization: 0.85
    maxModelLen: 32768  # Increase for longer context
    dtype: "auto"
    trustRemoteCode: true
    enablePrefixCaching: true
  
  persistence:
    enabled: true
    size: 50Gi  # Model cache storage
  
  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
    limits:
      memory: "32Gi"
      nvidia.com/gpu: 1
```

### Email Alerts Configuration

To enable email alerts, set these environment variables:

```yaml
env:
  SMTPLIB_SERVER: "smtp.gmail.com"
  SMTPLIB_PORT: "587"
  SMTPLIB_USERNAME: "your-email@gmail.com"
  SMTPLIB_PASSWORD: "your-app-password"
  SENDER_EMAIL: "your-email@gmail.com"
  ADMIN_EMAIL: "admin@example.com"
```

Or use a Kubernetes secret:

```bash
kubectl create secret generic camera-agent-secrets \
  --from-literal=SMTPLIB_PASSWORD=your-app-password
```

### Persistence

```yaml
persistence:
  data:
    enabled: true
    storageClass: ""  # Uses default storage class
    size: 10Gi        # App data, logs, detected images
    
vllmServer:
  persistence:
    enabled: true
    size: 50Gi        # Model cache storage
```

## Example Values Files

### Minimal Jetson Thor Deployment

```yaml
# values-jetson-minimal.yaml
image:
  tag: "v4"

camera:
  enabled: true
  devicePath: /dev/video0

vllmServer:
  enabled: true
  args:
    maxModelLen: 16384
    gpuMemoryUtilization: 0.80
```

### Production Deployment with Email

```yaml
# values-production.yaml
image:
  tag: "v4"
  pullPolicy: Always

camera:
  enabled: true
  devicePath: /dev/video0

vllmServer:
  enabled: true
  args:
    maxModelLen: 32768
    gpuMemoryUtilization: 0.85

env:
  LOG_LEVEL: "INFO"
  SMTPLIB_SERVER: "smtp.gmail.com"
  SMTPLIB_PORT: "587"
  SMTPLIB_USERNAME: "alerts@company.com"
  SENDER_EMAIL: "alerts@company.com"
  ADMIN_EMAIL: "ops@company.com"

persistence:
  data:
    enabled: true
    size: 20Gi
```

## Installation Examples

### Basic Installation

```bash
helm install camera-agent ./camera-agent
```

### With Custom Values File

```bash
helm install camera-agent ./camera-agent -f values-production.yaml
```

### Override Specific Values

```bash
helm install camera-agent ./camera-agent \
  --set image.tag=v4 \
  --set vllmServer.args.maxModelLen=32768 \
  --set env.ADMIN_EMAIL=ops@company.com
```

### Upgrade Existing Deployment

```bash
helm upgrade camera-agent ./camera-agent \
  --set image.tag=v4 \
  --reuse-values
```

## API Reference

### Web UI Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard - real-time monitoring status |
| `/monitoring` | Live camera feed and analysis |
| `/logs` | Detection event history |
| `/settings` | Configuration and alert settings |
| `/users` | User/recipient management |
| `/configuration` | Advanced YAML configuration |

### REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Readiness check |
| `GET` | `/api/status` | Monitoring status and metrics |
| `POST` | `/api/start_monitoring` | Start camera monitoring |
| `POST` | `/api/stop_monitoring` | Stop camera monitoring |
| `GET` | `/api/video_feed` | MJPEG video stream |
| `GET` | `/api/capture_frame` | Capture single frame |
| `POST` | `/api/analyze_prompt` | Analyze frame with custom prompt |
| `POST` | `/api/analyze_agentic` | Run agentic analysis with tool calling |
| `GET` | `/api/tools` | List available agent tools |
| `GET/POST` | `/api/config` | Get/update configuration |
| `GET` | `/api/config/defaults` | Get default configuration |
| `POST` | `/api/config/reset` | Reset to default configuration |
| `GET/PUT` | `/api/notifications/recipients` | Manage alert recipients |
| `GET/POST` | `/api/users` | User management |
| `GET/DELETE` | `/api/logs` | Detection logs |
| `POST` | `/api/logs/export` | Export logs (CSV/JSON) |
| `GET` | `/api/agent/memory` | Agent memory state |
| `GET/POST` | `/api/agent/prompt` | Detection prompt management |
| `GET` | `/api/test_camera` | Test camera connectivity |
| `GET` | `/api/test_inference` | Test VLM connectivity |
| `GET` | `/api/test_vllm` | Test vLLM server |
| `GET` | `/api/system/status` | System diagnostics |

## Monitoring & Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app.kubernetes.io/name=camera-agent
```

### View Logs

```bash
# Camera agent logs
kubectl logs -l app.kubernetes.io/name=camera-agent -c camera-agent -f

# vLLM server logs
kubectl logs -l app.kubernetes.io/component=vllm -f
```

### Check vLLM Health

```bash
# Get vLLM service ClusterIP
kubectl get svc camera-agent-vllm

# Port-forward to test locally
kubectl port-forward svc/camera-agent-vllm 8000:8000

# Test vLLM health
curl http://localhost:8000/health
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **vLLM takes long to start** | Model download can take 10-20 minutes on first run. Check logs for download progress. |
| **Camera not found** | Verify `/dev/video0` exists and is accessible. Check `securityContext.privileged: true`. |
| **GPU OOM** | Reduce `gpuMemoryUtilization` or `maxModelLen` in vLLM config. |
| **Circuit breaker open** | Use `/api/circuit_breaker/reset` or restart monitoring to recover from failures. |
| **No alerts sent** | Check email configuration and recipient list in Settings page. |

## Agent Tools

The agent supports agentic tool-calling for automated actions:

| Tool | Description |
|------|-------------|
| `send_alert_email` | Send email notifications with optional image attachments |
| `save_evidence` | Save detection images to disk for review |
| `log_event` | Record events to the persistent database |
| `notify_desktop` | Show desktop notification (when available) |
| `query_history` | Query recent detection history for context |
| `analyze_trend` | Analyze detection patterns over time |

## Uninstall

```bash
helm uninstall camera-agent

# Clean up PVCs if needed
kubectl delete pvc -l app.kubernetes.io/name=camera-agent
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                               │
│                                                                          │
│  ┌──────────────────────────┐      ┌──────────────────────────────────┐ │
│  │    Camera Agent Pod      │      │         vLLM Server Pod          │ │
│  │  ┌────────────────────┐  │      │  ┌────────────────────────────┐  │ │
│  │  │   Flask + SocketIO │  │ HTTP │  │  NVIDIA Triton + vLLM      │  │ │
│  │  │   Web Application  │──┼──────┤  │  Qwen3-VL-8B-Instruct      │  │ │
│  │  └────────────────────┘  │      │  │  (Vision Language Model)   │  │ │
│  │           │              │      │  └────────────────────────────┘  │ │
│  │  ┌────────┴────────┐     │      │             │                    │ │
│  │  │ Monitoring      │     │      │        NVIDIA GPU                │ │
│  │  │ Agent + Tools   │     │      └──────────────────────────────────┘ │
│  │  └─────────────────┘     │                                          │
│  │           │              │      ┌──────────────────────────────────┐ │
│  │      /dev/video0         │      │         Persistent Volumes       │ │
│  │       (Camera)           │      │  - Model cache (50Gi)            │ │
│  └──────────────────────────┘      │  - App data (10Gi)               │ │
│              │                     └──────────────────────────────────┘ │
│       NodePort:30080                                                    │
└──────────────┼──────────────────────────────────────────────────────────┘
               │
         ┌─────┴─────┐
         │  Browser  │
         │  (Web UI) │
         └───────────┘
```

## Detection Logic

1. **Frame Capture**: Captures frames from camera at configurable intervals
2. **SSIM Analysis**: Scene change detection reduces unnecessary VLM calls (~95% reduction)
3. **Vision Analysis**: Sends changed frames to Qwen3-VL for scene description
4. **Box Detection**: Identifies shipping/packaging boxes in the frame
5. **Label Verification**: Checks if boxes have visible shipping labels
6. **Alert Routing**: Sends notifications only for unlabeled boxes

## License

Apache 2.0
