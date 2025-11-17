# ZEDEDA Camera Monitoring Agent

An AI-assisted camera system that detects packaging/shipping boxes in real time and raises alerts only when a box is missing a visible shipping label.

## Overview

The ZEDEDA Camera Monitoring Agent captures video from `/dev/video0`, uses SSIM-based preprocessing to detect scene changes, analyzes frames with Ollama vision models, and differentiates between labeled and unlabeled packaging boxes. Alerts are generated only for boxes without clearly visible shipping labels, helping highlight items that may be misplaced or not yet processed.

## Key Features

- **Smart Frame Preprocessing**: SSIM-based scene change detection reduces LLM calls by ~95%
- **Agent Similarity Guard**: Reuses the last decision when frames are nearly identical, preventing redundant LLM calls
- **Two-Stage LLM Pipeline**: Vision model summarizes the scene, decision model classifies box/label status
- **Classical Packaging Detector**: Contour-based OpenCV analyzer scores packaging-shaped regions before any LLM call
- **Shipping Label Awareness**: Hybrid pipeline merges LLM judgment with a local OpenCV-based analyzer to verify label visibility
- **Tool Trace Logging**: Decision trace summaries record which LLM tools executed, visible in console logs and the web UI
- **Confidence Blending**: Weighted scoring fuses LLM confidence with classical packaging/label analyzers for transparent decision strength
- **Agent Memory & Summaries**: Maintains a rolling memory of recent events and surfaces quick summaries without additional LLM calls
- **Responsive Polling**: Processes frames immediately on scene changes, every 5s when static
- **Email & Desktop Notifications**: Automatic alerting with customizable templates
- **Frame Logging**: Saves processed frames and detection imagery for auditability

## Architecture

```
Camera Feed → SSIM Analysis → Vision LLM Summary → Decision LLM (label check) → Alert Routing
   ↓              ↓                 ↓                         ↓                  ↓
/dev/video0    Frame Diff      Ollama gemma3:4b         llama3.2 decision     Email / UI Alert
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

3. **Configure Recipients** (Settings UI or `config.yaml`):
   - Recommended: open the web app → **Settings → Alert Recipients** and add the desired email addresses.
   - Alternatively, update `notifications.email.recipients` (and the corresponding rule `to` lists) in `config.yaml`:

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

### Camera Settings (`config.yaml`)
```yaml
advanced:
   agent_ssim_skip_threshold: 0.95
   agent_ssim_recheck_seconds: 30
   motion_burst_interval: 0.2
   motion_burst_window: 12
camera:
   device_index: 0
   capture_interval: 5
   save_detection_images: true
   save_processed_frames: true
   preprocessing:
      diff_threshold: 0.80
ollama:
   url: "http://localhost:11434"
   vision_model: "gemma:12b"
   decision_model: "llama3.2:latest"
detection:
   prompt: |
      Describe what you see in this image in 2-3 concise sentences.
      Highlight any packaging or shipping boxes and mention whether shipping labels are clearly visible.
```

> Advanced SSIM tuning: adjust `advanced.agent_ssim_skip_threshold` (similarity to reuse), `advanced.agent_ssim_recheck_seconds` (max age before re-running the LLM), and `advanced.agent_ssim_reference_size` (width/height for SSIM downsampling) to fit your scene dynamics.

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
CAMERA_AGENT_CONFIG=/data/config.yaml  # optional override for container mounts
```

## Detection Logic

1. **Frame Capture**: Captures frames from the camera, prioritizing scene changes detected via SSIM
2. **Similarity Guard**: Compares the current frame against the last analyzed frame via SSIM and reuses the previous decision when similarity ≥ `agent_ssim_skip_threshold`
3. **Vision Summary**: Sends selected frames to `gemma3:4b` (or configured model) for a concise description
4. **Classical Packaging Analysis**: Applies a contour/geometry-based OpenCV model to score packaging-box candidates, generating interpretable hints
5. **Label Region Clustering**: Uses a k-means model over candidate regions to estimate how many distinct shipping labels appear per box and feeds those counts into downstream tooling
6. **Decision Pass**: Provides the description (plus local hints) to `llama3.2` for classification into `BOX_NO_LABEL`, `BOX_WITH_LABEL`, or `NO_BOX_DETECTED`
7. **Confidence Blending**: Combines the LLM's reported confidence with classical analyzer scores for a traceable final confidence metric
8. **Tool Call (Alerts)**: When `BOX_NO_LABEL` is confirmed, the decision model issues a structured tool call, including confidence, reasoning, and label status
9. **Alert Routing**: The agent sends notifications only if no shipping label is detected; labeled boxes are logged without alerting

## Directory Structure

```
├── camera_agent.py           # Main monitoring application
├── config.yaml               # Configuration file
├── .env                      # Environment variables
├── detected_images/          # Images where packaging boxes were detected
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
