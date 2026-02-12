# ZEDEDA Camera Monitoring Agent

An AI-powered camera monitoring system that inspects printed circuit boards (PCBs) on a conveyor and raises alerts for defect conditions.

## Overview

The ZEDEDA Camera Monitoring Agent captures video from a camera device and now features a proactive, LLM-directed monitoring loop. The agent continuously observes the conveyor feed, maintains temporal context, reasons about PCB motion and stability, and decides when to run lightweight checks versus full defect inspections. Alerts are generated only when the LLM concludes that the user’s natural-language instruction requires action (e.g., a PCB defect).

## Key Features

- **Proactive Monitoring Loop**: Two-stage LLM pipeline (observation + decision) keeps temporal context and autonomously pulls the trigger on analyses.
- **Instruction-Aware Reasoning**: User prompts like “watch for PCB defects” or “inspect stopped boards under the camera” guide the agent’s behavior without code changes.
- **Adaptive Actions**: LLM chooses between wait, quick_check, and full_inspection, eliminating rigid SSIM/threshold heuristics.
- **Agent Memory & Summaries**: Rolling memory plus scene signatures prevent redundant inspections of the same object.
- **Agentic Tooling & Alerts**: Full inspections reuse the Unified VLM + alert stack (email, WebSocket, DB logging).
- **Real-time Dashboard**: Flask web UI with live feed, logs, and proactive status snapshots.

## Architecture

```
Camera Feed → SSIM Analysis → vLLM (Qwen3-VL) → Tool Execution → Alert Routing
   ↓              ↓                  ↓                 ↓              ↓
/dev/video0    Frame Diff     Vision Analysis     send_email      Email / UI
```

## Proactive Monitoring Loop

The proactive agent runs a continuous loop driven entirely by the LLM:

1. **Observe** – Lightweight prompt asks the VLM to describe the scene, detect motion, and emit a stable `scene_signature`.
2. **Decide** – A second prompt consumes temporal context (user intent, last action, stability counters, inspection history) and selects `wait`, `quick_check`, or `full_inspection`.
3. **Act** – `quick_check` performs a cheap confirmation; `full_inspection` reuses the Vision-Language analysis/alert stack; `wait` keeps monitoring.

Start the agent with a natural-language instruction via REST:

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
   -H "Content-Type: application/json" \
   -d '{
            "instruction": "Monitor the conveyor for PCB defects",
            "frame_interval": 1.2,
            "stability_frames": 5,
            "decision_temperature": 0.2
         }'
```

Check status or stop the loop:

```bash
curl http://localhost:8080/api/monitoring/proactive/status
curl -X POST http://localhost:8080/api/monitoring/proactive/stop
```

### Project Structure

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
│   ├── vlm_service.py        # VLM client factory
│   └── config_service.py     # Config management
│
├── agents/                   # ML/AI components
│   ├── camera_agent.py       # Main monitoring agent
│   ├── state.py              # Agent memory
│   ├── alerting.py           # Alert manager
│   ├── tools.py              # Tool executor
│   └── vlm/                  # Vision Language Model
│       ├── client.py         # Unified VLM client
│       └── prompts.py        # System prompts
│
├── templates/                # Jinja2 HTML templates
├── static/                   # CSS/JS assets
└── helm/                     # Kubernetes Helm charts
```

## Quick Start

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start vLLM Server** (requires GPU):
   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000
   ```

3. **Configure Environment** (`.env`):
   ```env
   VLLM_URL=http://localhost:8000
   VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASS=your-app-password
   ```

4. **Run**:
   ```bash
   python run.py
   ```

5. **Open Dashboard**: http://localhost:8080

### Docker Compose

```bash
docker-compose up -d
```

This starts both the camera-agent and vLLM server containers.

### Kubernetes (Helm)

```bash
helm install camera-agent ./helm/camera-agent
```

The Helm chart deploys:
- Camera-agent pod (Flask web app)
- vLLM server pod (GPU inference)
- Persistent storage for models and data

## Configuration

### config.yaml
```yaml
camera:
  device_index: 0
  capture_interval: 30
  save_detection_images: true

vllm:
  url: http://localhost:8000
  model: Qwen/Qwen3-VL-8B-Instruct
  timeout: 300
  temperature: 0.1

advanced:
  agent_ssim_skip_threshold: 0.95
  agent_ssim_recheck_seconds: 30

notifications:
  email:
    recipients:
      - admin@company.com
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM server URL | `http://localhost:8000` |
| `VISION_MODEL` | Model name | `Qwen/Qwen3-VL-8B-Instruct` |
| `CAMERA_INDEX` | Camera device index | `0` |
| `EMAIL_USER` | SMTP username | - |
| `EMAIL_PASS` | SMTP password | - |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Get configuration |
| `POST` | `/api/config` | Update configuration |
| `GET` | `/api/monitoring/status` | Monitoring status |
| `GET` | `/api/monitoring/proactive/status` | Proactive agent snapshot |
| `POST` | `/api/monitoring/proactive/start` | Start/update proactive monitoring |
| `POST` | `/api/monitoring/proactive/stop` | Stop proactive monitoring |
| `POST` | `/api/analysis/analyze` | Analyze single frame |
| `GET` | `/api/logs` | Detection logs |

## WebSocket Events

- `connect` - Client connected
- `status_update` - Monitoring status changed
- `new_detection` - New detection event
- `frame_update` - Live frame preview

## Testing

```bash
pytest tests/ -v
```

## Dependencies

- **Web**: Flask, Flask-SocketIO, gunicorn
- **Vision**: opencv-python, scikit-image, numpy
- **Config**: PyYAML, python-dotenv, pydantic

## License

ZEDEDA Proprietary Software
