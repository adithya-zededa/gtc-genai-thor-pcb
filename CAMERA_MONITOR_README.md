# ZEDEDA Camera Monitor Detection Agent

An intelligent camera monitoring agent that uses Ollama's LLaVA vision model to detect computer monitors in real-time camera feeds and send alerts when detected.

## Features

- 🎥 **Real-time Camera Monitoring** - Continuously monitors `/dev/video0` camera feed
- 🤖 **AI-Powered Detection** - Uses Ollama LLaVA vision model for computer monitor detection
- 📧 **Smart Alerts** - Configurable email notifications when monitors are detected
- 🔧 **Flexible Configuration** - YAML-based configuration for rules and settings
- 🐳 **Container Ready** - Docker and Docker Compose support
- 🖥️ **System Service** - Systemd service for production deployment
- 📊 **Statistics Tracking** - Built-in monitoring and logging

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the project
cd /home/nvidia/Developer/agent-attempt-2

# Create environment configuration
./start_camera_agent.sh setup

# Edit the .env file with your settings
nano .env
```

### 2. Configure Email Alerts

Update the `.env` file with your email settings:

```bash
# Email configuration
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
```

### 3. Test the Setup

```bash
# Run system checks
./start_camera_agent.sh check

# Run comprehensive test
python3 test_camera_setup.py
```

### 4. Start Monitoring

```bash
# Start the camera agent
./start_camera_agent.sh start
```

## Configuration

The agent is configured via `camera_config.yaml`:

### Camera Settings
```yaml
camera:
  device_index: 0              # /dev/video0
  capture_interval: 5          # seconds between captures
  save_detection_images: true  # save images when monitors detected
```

### Detection Settings
```yaml
detection:
  prompt: |
    Look carefully at this image. Is there a computer monitor, 
    laptop screen, or any kind of display screen visible?
  confidence_threshold: 0.7
  positive_keywords: ["monitor", "screen", "display", "computer", "laptop"]
```

### Alert Rules
```yaml
rules:
  - id: "MONITOR_DETECTED"
    description: "Trigger alert when a computer monitor is detected"
    enabled: true
    email:
      to: ["admin@zededa.com"]
      subject: "🖥️ Computer Monitor Detected"
      body: |
        A computer monitor has been detected at {{timestamp}}
        Confidence: {{confidence}}
        Details: {{full_response}}
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View camera agent logs
docker-compose logs -f camera-agent

# Stop services
docker-compose down
```

### Option 2: Systemd Service

```bash
# Install as system service
./start_camera_agent.sh service

# Start the service
sudo systemctl start zededa-camera-agent

# Check status
sudo systemctl status zededa-camera-agent

# View logs
sudo journalctl -u zededa-camera-agent -f
```

### Option 3: Direct Python Execution

```bash
# Run directly
python3 camera_agent.py --config camera_config.yaml

# Run with test mode
python3 camera_agent.py --test
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `VISION_MODEL` | `llava:7b` | Vision model to use |
| `CAMERA_INDEX` | `0` | Camera device index (/dev/video0) |
| `CAPTURE_INTERVAL` | `5` | Seconds between captures |
| `EMAIL_USER` | - | Email username for alerts |
| `EMAIL_PASS` | - | Email password for alerts |
| `LOG_LEVEL` | `INFO` | Logging level |

## Requirements

### System Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- Camera device at `/dev/video0`
- Ollama server with LLaVA model

### Python Dependencies
```bash
pip install opencv-python PyYAML python-dotenv requests plyer
```

### Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull vision model
ollama pull llava:7b

# Start Ollama server
ollama serve
```

## Testing

### Quick Test
```bash
python3 test_camera_setup.py
```

### Full System Check
```bash
./start_camera_agent.sh check
```

### Test Individual Components
```bash
# Test camera access
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed'); cap.release()"

# Test Ollama connection
curl -s http://localhost:11434/api/version

# Test configuration
python3 camera_agent.py --test
```

## Troubleshooting

### Camera Issues

**Camera not found:**
```bash
# List available cameras
ls -la /dev/video*

# Check permissions
sudo chmod 666 /dev/video0

# Add user to video group
sudo usermod -a -G video $USER
```

**Permission denied:**
```bash
# Fix camera permissions
sudo chmod 666 /dev/video0

# Or add udev rule for persistent permissions
echo 'KERNEL=="video[0-9]*", GROUP="video", MODE="0666"' | sudo tee /etc/udev/rules.d/99-camera.rules
sudo udevadm control --reload-rules
```

### Ollama Issues

**Connection failed:**
```bash
# Check if Ollama is running
curl -s http://localhost:11434/api/version

# Start Ollama
ollama serve

# Check available models
ollama list
```

**Model not found:**
```bash
# Pull the vision model
ollama pull llava:7b

# Or use a different model
export VISION_MODEL=llava:13b
```

### Email Issues

**Authentication failed:**
- Use app-specific passwords for Gmail
- Enable 2FA and generate app password
- Check SMTP settings for your provider

**Emails not sending:**
```bash
# Test email configuration
python3 -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'your-app-password')
print('Email config OK')
"
```

## Monitoring and Logs

### View Logs
```bash
# Direct execution logs
tail -f camera_agent.log

# Docker logs
docker-compose logs -f camera-agent

# Systemd logs
sudo journalctl -u zededa-camera-agent -f
```

### Statistics
The agent tracks:
- Total frames processed
- Number of detections
- Alerts sent
- Processing times
- Error rates

### Performance Tuning

**Reduce CPU usage:**
- Increase `capture_interval` (e.g., 10 seconds)
- Lower camera resolution in config
- Use lighter vision model

**Improve accuracy:**
- Customize detection prompt
- Adjust confidence threshold
- Use larger vision model (llava:13b)

## Integration

### Custom Rules
Add custom detection rules in `camera_config.yaml`:

```yaml
rules:
  - id: "LAPTOP_DETECTED"
    description: "Detect laptop screens specifically"
    enabled: true
    conditions:
      - field: "full_response"
        operator: "contains"
        value: "laptop"
    email:
      to: ["security@company.com"]
      subject: "Laptop Screen Detected"
```

### Webhook Integration
Extend the agent to send webhooks:

```python
# Add to AlertManager class
def send_webhook(self, event, webhook_url):
    payload = {
        "timestamp": event.timestamp,
        "type": "monitor_detected",
        "confidence": event.confidence,
        "response": event.full_response
    }
    requests.post(webhook_url, json=payload)
```

### External Monitoring
Monitor agent health via:
- Log file parsing
- HTTP health endpoint (future enhancement)
- Process monitoring tools

## Security

### Recommendations
- Run as non-root user
- Limit camera access permissions
- Secure email credentials
- Use TLS for Ollama communication
- Regular security updates

### Privacy
- Detection images are saved locally only
- No data sent to external services (except Ollama)
- Email alerts contain minimal information
- Configure retention policies for logs/images

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Test individual components
4. Verify configuration settings

## License

This project is part of the ZEDEDA AI Agent suite and follows the same licensing terms.