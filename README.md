# ZEDEDA Camera Monitoring Agent

An AI-powered camera monitoring system that inspects printed circuit boards (PCBs) on a conveyor and raises alerts for defect conditions. Designed to run on **NVIDIA Jetson Thor** with the **Cosmos Reason 2 8B** vision-language model.

## Overview

The ZEDEDA Camera Monitoring Agent captures video from a camera device and runs a deterministic computer-vision observation loop that watches the conveyor feed for motion, board presence, and board identity. When a board stops in the inspection zone, the loop hands off to an LLM-driven agent that performs the actual inspection via MCP tool calls. Alerts are generated only when the LLM concludes that the user's natural-language instruction requires action (e.g., a PCB defect).

## Key Features

- **Deterministic Monitoring Loop**: Pure CV observation (frame-diff motion, board segmentation, presence debouncing, IoU tracking) detects when a board stops in the inspection zone and fires a single `on_board_ready` callback — no CV-side judgment calls.
- **Instruction-Aware Reasoning**: User prompts like "watch for PCB defects" or "inspect stopped boards under the camera" guide agent behavior without code changes.
- **Agentic Tool Calling**: The VLM decides what to do next — inspect, classify, alert, log — by calling MCP tools directly, eliminating rigid threshold heuristics.
- **Agent Memory & Summaries**: Rolling memory plus scene signatures prevent redundant inspections.
- **Agentic Tooling & Alerts**: Full inspections use the Unified VLM + alert stack (email, WebSocket, DB logging).
- **Real-time Dashboard**: Flask web UI with live feed, logs, and proactive status snapshots.

---

## Hardware & Software Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| **Device** | NVIDIA Jetson Thor |
| **GPU** | Integrated NVIDIA GPU with ≥16 GB VRAM |
| **RAM** | 32 GB+ recommended |
| **Storage** | 100 GB+ (model weights ~16 GB, plus container images and data) |
| **Camera** | USB camera mounted at `/dev/video0` |

### Software

| Component | Version / Details |
|-----------|-------------------|
| **JetPack** | 7+ (BSP for Jetson Thor) |
| **NVIDIA Container Runtime** | Included with JetPack — provides GPU access inside containers |
| **Docker** | 24.0+ (included with JetPack, or install separately) |
| **K3s** | v1.31+ (lightweight Kubernetes for edge) |
| **Helm** | v3.14+ |
| **NVIDIA GPU Device Plugin** | For Kubernetes GPU scheduling |
| **Python** | 3.11 (if running outside containers) |

### Default Model

The default vision-language model is **`nvidia/Cosmos-Reason2-8B`** served via the NVIDIA Triton Server with vLLM backend (`nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3`).

> For gated models on Hugging Face, you will need a Hugging Face access token.

---

## Installation — Step by Step

### 1. Install Docker

Docker should already be installed with JetPack. Verify:

```bash
docker --version
```

If not installed:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

### 2. Install the NVIDIA Container Runtime

This is included with JetPack, but confirm it is configured as the default Docker runtime:

```bash
# Verify nvidia runtime is available
docker info | grep -i runtime
```

You should see `nvidia` listed. If not, install it:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Set NVIDIA as the **default runtime** (required for K3s). Edit `/etc/docker/daemon.json`:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

### 3. Install K3s (Lightweight Kubernetes)

K3s is the recommended Kubernetes distribution for Jetson edge devices.

```bash
# Install K3s with Docker as the container runtime
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker" sh -

# Verify the node is ready
sudo k3s kubectl get nodes
```

Set up kubeconfig for non-root use:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config
# Add to your shell profile:
echo 'export KUBECONFIG=~/.kube/config' >> ~/.bashrc
```

Verify kubectl works:

```bash
kubectl get nodes
# Should show your node in "Ready" status
```

### 4. Install the NVIDIA GPU Device Plugin for Kubernetes

The device plugin exposes GPUs to Kubernetes so pods can request `nvidia.com/gpu` resources.

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
```

Verify GPUs are visible to the cluster:

```bash
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
# Should output "1" (or more)
```

### 5. Install Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

---

## Running with Helm (Recommended for Production)

### Basic Install

```bash
cd /path/to/gtc-genai-thor-pcb

helm install camera-agent ./helm/camera-agent
```

This deploys:
- **Camera-agent pod** — Flask web app with proactive monitoring loop
- **vLLM server pod** — GPU inference using `nvidia/Cosmos-Reason2-8B`
- **Persistent volumes** — For model cache and application data

### Install with a Hugging Face Token (for Gated Models)

```bash
helm install camera-agent ./helm/camera-agent \
  --set vllmServer.huggingfaceToken=hf_YOUR_TOKEN_HERE
```

Or pre-create a Kubernetes secret:

```bash
kubectl create secret generic hf-tokens --from-literal=hf-token=hf_YOUR_TOKEN_HERE

helm install camera-agent ./helm/camera-agent \
  --set secrets.existingSecret=hf-tokens
```

### Common Overrides

```bash
helm install camera-agent ./helm/camera-agent \
  --set vllmServer.model="nvidia/Cosmos-Reason2-8B" \
  --set vllmServer.args.gpuMemoryUtilization=0.5 \
  --set vllmServer.args.maxModelLen=16000 \
  --set camera.devicePath=/dev/video0
```

### Check Deployment Status

```bash
# Watch pods come up
kubectl get pods -w

# Check vLLM logs (model download + loading can take 10+ minutes the first time)
kubectl logs -f deployment/camera-agent-vllm -c vllm

# Check camera-agent logs
kubectl logs -f deployment/camera-agent
```

### Access the Dashboard

The service is exposed as a NodePort on **port 30080** by default:

```
http://<JETSON_IP>:30080
```

### Upgrade / Uninstall

```bash
# Upgrade with new values
helm upgrade camera-agent ./helm/camera-agent --set vllmServer.args.gpuMemoryUtilization=0.7

# Uninstall
helm uninstall camera-agent
```

### Helm Values Reference

Key values in `helm/camera-agent/values.yaml`:

| Value | Default | Description |
|-------|---------|-------------|
| `vllmServer.enabled` | `true` | Deploy vLLM server alongside camera-agent |
| `vllmServer.model` | `nvidia/Cosmos-Reason2-8B` | VLM model to serve |
| `vllmServer.image.tag` | `25.12-vllm-python-py3` | Triton + vLLM container tag |
| `vllmServer.args.gpuMemoryUtilization` | `0.5` | Fraction of GPU VRAM to use |
| `vllmServer.args.maxModelLen` | `16000` | Max context length (tokens) |
| `vllmServer.runtimeClassName` | `nvidia` | Kubernetes runtime class for GPU |
| `camera.enabled` | `true` | Mount camera device into pod |
| `camera.devicePath` | `/dev/video0` | Host camera device path |
| `service.nodePort` | `30080` | NodePort for web dashboard |
| `persistence.data.size` | `10Gi` | PVC size for app data |
| `vllmServer.persistence.size` | `50Gi` | PVC size for model cache |

---

## Running with Docker Compose (Development)

Docker Compose is useful for local development and testing. The compose file starts both the camera-agent and a vLLM server.

```bash
cd /path/to/gtc-genai-thor-pcb

docker-compose up -d
```

> **Note:** The compose file uses `Qwen/Qwen3-VL-4B-Instruct` by default for lighter resource usage during development. Edit `docker-compose.yml` to change the model.

Access the dashboard at **http://localhost:8080**.

---

## Running Locally (Development)

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start a vLLM server** (requires GPU):
   ```bash
   vllm serve nvidia/Cosmos-Reason2-8B --host 0.0.0.0 --port 8000 \
     --gpu-memory-utilization 0.5 --max-model-len 16000 --trust-remote-code
   ```

3. **Configure environment** (`.env` or export):
   ```bash
   export VLLM_URL=http://localhost:8000
   export CAMERA_INDEX=0
   ```

4. **Run the app:**
   ```bash
   python run.py
   ```

5. **Open Dashboard:** http://localhost:8080

---

## Architecture

```
Camera Feed → CV Gating → VLM (Cosmos Reason 2 8B) → Tool Execution → Alert Routing
   ↓              ↓                  ↓                      ↓              ↓
/dev/video0   Motion/Zone     Vision Analysis          send_email      Email / UI
              Detection       & Reasoning              log_event       WebSocket
```

## Proactive Monitoring Loop

`MonitoringLoop` (`agents/core/monitoring_loop.py`) is the only deterministic component in the agent system. It never decides *what* to do — only *when* something needs attention:

1. **Observe** – Frames are pulled from the camera publisher and scored with pure CV sensors: frame-diff motion score, edge density, board segmentation, presence debouncing, and IoU-based board tracking. These are sensor readings, not decisions.
2. **Detect a state change** – When a tracked board transitions from moving to stopped inside the inspection zone (and its signature hasn't already been inspected), the loop waits ~4 seconds for the camera to auto-focus, grabs a fresh frame, and fires a single `on_board_ready(frame, context)` callback.
3. **Decide & act** – The callback owner invokes the LLM-driven inspection (`StreamlinedAgent.analyze_agentic` in `agents/core/detection_agent.py`), which uses `UnifiedVLMClient.analyze_with_tools` (`agents/vlm/client.py`) to let the vision-language model call MCP tools — `inspect_pcb`, `classify_board`, `send_defect_alert`, `log_defect`, and more — against the MCP tool layer. There is no separate `wait` / `quick_check` / `full_inspection` selection step anymore; every board-ready event goes straight to a single LLM-driven inspection.

### Start Proactive Monitoring via API

```bash
curl -X POST http://<JETSON_IP>:30080/api/monitoring/proactive/start \
   -H "Content-Type: application/json" \
   -d '{
         "instruction": "Monitor the conveyor for PCB defects",
         "frame_interval": 1.2,
         "stability_frames": 5,
         "decision_temperature": 0.2
       }'
```

### Check / Stop

```bash
curl http://<JETSON_IP>:30080/api/monitoring/proactive/status
curl -X POST http://<JETSON_IP>:30080/api/monitoring/proactive/stop
```

---

## Project Structure

```
├── run.py                    # Main entry point
├── wsgi.py                   # Production WSGI entry
├── config.yaml               # Configuration file
│
├── core/                     # Core infrastructure
│   ├── config.py             # Configuration management
│   ├── logging.py            # Logging setup
│   └── utils.py              # Utility functions
│
├── app/                      # Flask application
│   ├── __init__.py           # Application factory
│   ├── api/v1/               # REST API endpoints
│   ├── views/                # HTML template routes
│   ├── websocket/            # WebSocket handlers
│   └── database/             # SQLite database layer
│
├── services/                 # Business logic
│   ├── camera_service.py     # Camera frame publisher
│   ├── monitoring_service.py # Monitoring orchestration
│   └── config_service.py     # Config management
│
├── agents/                   # AI agent components
│   ├── core/
│   │   ├── monitoring_loop.py  # CV observation loop
│   │   ├── detection_agent.py  # VLM detection/analysis agent
│   │   └── state.py            # Agent memory & DetectionEvent
│   ├── tools/                  # Tool executor (email, alerts)
│   └── vlm/                    # Vision Language Model
│       ├── client.py           # Unified VLM client
│       └── prompts.py          # System prompts
│
├── templates/                # Jinja2 HTML templates
├── static/                   # CSS/JS assets
├── helm/                     # Kubernetes Helm charts
│   └── camera-agent/
│       ├── Chart.yaml
│       ├── values.yaml       # All configurable values
│       └── templates/        # K8s manifests
└── docker-compose.yml        # Local dev compose
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Get configuration |
| `POST` | `/api/config` | Update configuration |
| `GET` | `/api/status` | Comprehensive system status (circuit breaker, agent stats) |
| `GET` | `/api/monitoring/status` | Dashboard monitoring overview — agent state + defect counts (`app/api/v1/defects.py`) |
| `GET` | `/api/monitoring/proactive/status` | Proactive agent snapshot |
| `POST` | `/api/monitoring/proactive/start` | Start/update proactive monitoring |
| `POST` | `/api/monitoring/proactive/stop` | Stop proactive monitoring |
| `POST` | `/api/analyze_prompt` | Analyze the current camera frame with a dynamic prompt |
| `POST` | `/api/analyze_agentic` | Analyze the current frame with agentic MCP tool calling |
| `POST` | `/api/analyze_uploaded_image` | Analyze an uploaded image (agentic or single-shot) |
| `GET` | `/api/defects/summary` | Defect summary for a time window |
| `GET` | `/api/tools` | List available MCP tools |
| `GET` | `/api/logs` | Detection logs |

> This is a quick-reference subset. See [`docs/api/endpoints.md`](docs/api/endpoints.md) for the full, exhaustive endpoint list (analysis, camera, config, defects, health, LLM, logs, MCP, monitoring, system, users).

## WebSocket Events

Socket.IO events are handled in `app/websocket/chat.py` and `app/websocket/__init__.py`.

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | client → server | Client connects; auto-initializes the chat session |
| `chat_connected` | server → client | Session bootstrap payload (agent state, tools, history) |
| `chat_message` | both | Submit user input / echo user, assistant, and tool messages |
| `agent_activity` | server → client | Real-time activity chips (in-progress/completed/failed) |
| `agent_state_changed` | server → client | Broadcast agent state transitions |
| `tool_confirmation_required` | server → client | A proposed tool call needs user approval |
| `detection_event` / `chat_detection` | server → client | Detection broadcast (generic / rendered as a chat message) |
| `frame_update` | server → client | Live frame preview (base64 image + metadata) |
| `monitoring_status` | server → client | Monitoring status update |
| `alert` | server → client | Alert broadcast |

> See [`docs/api/endpoints.md`](docs/api/endpoints.md) for the full event list, including the proposal-approval workflow and conversation utility events.

## Configuration

See `config.yaml` for the full configuration reference. Key sections:

| Section | Description |
|---------|-------------|
| `camera` | Camera device index, capture interval, image saving |
| `vllm` | vLLM server URL, model, timeout, temperature |
| `proactive` | Proactive loop tuning — frame interval, stability, temperatures |
| `advanced` | Legacy reactive mode settings |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM server URL | `http://localhost:8000` |
| `VISION_MODEL` | Model name (auto-detected from server if empty) | auto |
| `CAMERA_INDEX` | Camera device index | `0` |
| `EMAIL_USER` | SMTP username for alerts | — |
| `EMAIL_PASS` | SMTP password for alerts | — |

## Troubleshooting

### vLLM pod stuck in CrashLoopBackOff
- Check logs: `kubectl logs deployment/camera-agent-vllm -c vllm`
- The model download can take 10+ minutes on first run. The liveness probe waits 10 minutes before failing.
- Ensure enough disk space for model weights (~16 GB).
- Verify GPU is visible: `kubectl exec deployment/camera-agent-vllm -- nvidia-smi`

### No GPU visible in Kubernetes
- Verify the NVIDIA device plugin is running: `kubectl get pods -n kube-system | grep nvidia`
- Check that Docker's default runtime is `nvidia` (required for K3s).
- Restart K3s after changing Docker config: `sudo systemctl restart k3s`

### Camera not accessible
- Verify the camera device exists: `ls -l /dev/video0`
- The pod runs as privileged and maps `/dev/video0`. If your camera is on a different device, override `camera.devicePath` in helm values.

### Dashboard not loading
- Check NodePort: `kubectl get svc | grep camera-agent`
- Default is port `30080`. Access at `http://<JETSON_IP>:30080`.

## Testing

```bash
pytest tests/ -v
```

## License

ZEDEDA Proprietary Software
