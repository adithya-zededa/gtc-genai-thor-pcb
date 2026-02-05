# ZEDEDA Camera Monitoring Agent

An AI-powered camera monitoring system that detects packaging/shipping boxes in real time and raises alerts when a box is missing a visible shipping label.

## Overview

The ZEDEDA Camera Monitoring Agent captures video from a camera device, uses SSIM-based preprocessing to detect scene changes, analyzes frames with vision language models (vLLM), and differentiates between labeled and unlabeled packaging boxes. Alerts are generated only for boxes without clearly visible shipping labels.

## Key Features

- **Smart Frame Preprocessing**: SSIM-based scene change detection reduces VLM calls by ~95%
- **Agent Similarity Guard**: Reuses the last decision when frames are nearly identical
- **Vision Language Model**: Uses Qwen3-VL via vLLM for accurate image analysis
- **Tool Calling**: Agentic VLM with structured tool calls for email alerts
- **Agent Memory & Summaries**: Maintains rolling memory of recent events
- **Real-time Dashboard**: Flask web UI with WebSocket live updates
- **Email Notifications**: Automatic alerting with customizable templates

## Architecture

```
Camera Feed → SSIM Analysis → vLLM (Qwen3-VL) → Tool Execution → Alert Routing
   ↓              ↓                  ↓                 ↓              ↓
/dev/video0    Frame Diff     Vision Analysis     send_email      Email / UI
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
| `POST` | `/api/monitoring/start` | Start monitoring |
| `POST` | `/api/monitoring/stop` | Stop monitoring |
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
