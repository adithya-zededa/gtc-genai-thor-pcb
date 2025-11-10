# ZEDEDA Camera Monitoring Agent

A smart camera monitoring system that uses computer vision and AI to detect computer monitors in real-time and send email alerts.

## Overview

The ZEDEDA Camera Monitoring Agent captures video from `/dev/video0`, uses SSIM-based preprocessing to detect scene changes, analyzes frames with Ollama vision models, and sends email alerts when computer monitors are detected in the camera feed.

## Key Features

- **Smart Frame Preprocessing**: SSIM-based scene change detection reduces LLM calls by 95%
- **Intelligent Scene Analysis**: Uses gemma3:4b model to describe scenes naturally
- **Keyword-Based Detection**: Triggers alerts when scene descriptions contain "monitor" or related terms
- **Responsive Polling**: Processes frames immediately on scene changes, every 5s when static
- **Email Notifications**: Automatic alerts via Gmail SMTP with App Password authentication
- **Frame Logging**: Saves all processed frames and detection images for review

## Architecture

```
Camera Feed → SSIM Analysis → Scene Description → Keyword Detection → Email Alert
     ↓              ↓               ↓                    ↓              ↓
/dev/video0    Frame Diff      Ollama gemma3:4b    "monitor" found   Gmail SMTP
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install opencv-python requests python-dotenv pyyaml scikit-image
   ```

2. **Configure Environment** (create `.env`):
   ```env
   VISION_MODEL=gemma3:4b
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASS=your-app-password
   EMAIL_FROM=your-email@gmail.com
   ```

3. **Configure Recipients** (Settings UI or `camera_config.yaml`):
   - Recommended: open the web app → **Settings → Alert Recipients** and add the desired email addresses.
   - Alternatively, update `notifications.email.recipients` (and the corresponding rule `to` lists) in `camera_config.yaml`:

     ```yaml
     notifications:
       email:
         recipients:
           - admin@company.com
     ```

4. **Run**:
   ```bash
   python camera_agent.py              # Start monitoring
   python camera_agent.py --test       # Test setup
   ```

## Configuration

### Camera Settings (`camera_config.yaml`)
```yaml
camera:
  device_index: 0                    # /dev/video0
  capture_interval: 5                # Seconds between captures
  save_detection_images: true        # Save positive detections
  save_processed_frames: true        # Save all analyzed frames
  preprocessing:
    enabled: true
    diff_threshold: 0.80             # SSIM threshold (lower = more sensitive)

ollama:
  model: "gemma3:4b"                # Vision model
  base_url: "http://localhost:11434"

detection:
  prompt: |
    Describe what you see in this image in 2-3 concise sentences.
    Focus on identifying objects, furniture, electronics, and the general setting.
    Be specific about any computer equipment, screens, displays, or technology.
```

> Tip: Use the configuration page or `POST /api/config/reset` to restore the default YAML at any time. The API also exposes `GET /api/config/defaults` for read-only inspection.

### Email Configuration (`.env`)
```env
# Ollama Settings
VISION_MODEL=gemma3:4b
OLLAMA_URL=http://localhost:11434

# Email Configuration (secrets supplied via environment)
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-gmail-app-password
EMAIL_FROM=your-email@gmail.com
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
CAMERA_AGENT_CONFIG=/data/camera_config.yaml  # optional override for container mounts
```

## Detection Logic

1. **Frame Capture**: Captures frames from camera every 5 seconds
2. **SSIM Analysis**: Compares frames using Structural Similarity Index
3. **Scene Description**: Sends significantly different frames to gemma3:4b for natural language description
4. **Keyword Detection**: Checks if description contains monitor-related keywords:
   - "monitor", "monitors", "computer monitor", "desktop monitor"
   - "screen", "display", "computer screen", "laptop screen"
5. **Alert Trigger**: Sends email if keywords detected in scene description

## Directory Structure

```
├── camera_agent.py           # Main monitoring application
├── camera_config.yaml        # Configuration file
├── .env                      # Environment variables
├── detected_images/          # Images where monitors were detected
├── processed_frames/         # All frames sent to LLM for analysis
└── camera_agent.log         # Application logs
```

## Deployment Options

### Systemd Service
```bash
sudo cp zededa-security-agent.service /etc/systemd/system/
sudo systemctl enable zededa-security-agent
sudo systemctl start zededa-security-agent
```

### Docker
```bash
docker build -f Dockerfile.web -t zededa-camera-agent .
docker run -d --device=/dev/video0 zededa-camera-agent
```

### Kubernetes
```bash
helm install camera-agent ./helm/zededa-ai-agent/
```

## Performance Metrics

- **Efficiency**: ~96% reduction in LLM calls through SSIM preprocessing
- **Responsiveness**: Immediate processing on scene changes (SSIM < 0.80)
- **Accuracy**: Natural language descriptions provide better context than YES/NO logic
- **Reliability**: Processes ~4% of captured frames while maintaining full detection capability

## API Endpoints (Web Service)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/generate` | Direct Ollama model access |
| GET | `/` | Service information |

## Troubleshooting

### Common Issues

1. **Camera Access**: Ensure `/dev/video0` permissions and no other applications using camera
2. **Ollama Connection**: Verify Ollama service running on `localhost:11434`
3. **Email Authentication**: Use Gmail App Password, not regular password
4. **Model Download**: `ollama pull gemma3:4b` if model not available

### Debug Commands
```bash
# Test camera access
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.read()[0] else 'FAIL')"

# Test Ollama connection
curl http://localhost:11434/api/tags

# Test email configuration
python camera_agent.py --test
```

## Dependencies

- **Core**: opencv-python, requests, python-dotenv, pyyaml
- **Vision**: scikit-image (for SSIM calculations)
- **Optional**: plyer (desktop notifications)

## License

ZEDEDA Proprietary Software
| `GET`  | `/api/version`      | Proxy to `OLLAMA_URL/api/version`                               |
| `POST` | `/api/generate`     | Execute a text or multi-modal generation request against Ollama |
| `POST` | `/api/agent/run`    | Run the higher-level security agent workflow                    |
| `GET`  | `/api/system-health`| Memory, disk, and Ollama connectivity diagnostics               |

### `/api/generate`

Request body:

```json
{
   "prompt": "Summarise this log",
   "model": "llama3.1",            // optional
   "stream": false,                 // optional
   "images": ["<base64>"]          // optional, pass-through to Ollama
}
```

Response mirrors Ollama's `generate` output and always includes the model used.

### `/api/agent/run`

Request body is passed directly to the autonomous agent:

```json
{
   "subject": "Door opened",
   "body": "Badge 9981 used at 03:42",
   "context": {"location": "HQ"}
}
```

Response:

```json
{
   "output": "Escalation not required.",
   "raw": { ... full agent response ... }
}
```

## Health & Diagnostics

- `GET /health` returns `200` when Ollama is reachable; otherwise `502`
- `GET /api/system-health` reports aggregated component status (`ollama`, `memory`, `disk`) with an overall state

## Helm & Kubernetes

The Kubernetes Helm chart under `helm/zededa-ai-agent` deploys the Flask service alongside an Ollama instance. Update `values.yaml` to point to the desired container tag and resource requirements before running:

```bash
helm upgrade --install email-agent ./helm/zededa-ai-agent
```

## Development Notes

- The Flask app is production-ready by running under Gunicorn (`Dockerfile.web`)
- All logging goes through Python's standard logging module (`LOG_LEVEL` respected)
- Integration tests can call the API endpoints directly; there is no browser dependency anymore

## License

Refer to the ZEDEDA licensing terms included with this repository.
