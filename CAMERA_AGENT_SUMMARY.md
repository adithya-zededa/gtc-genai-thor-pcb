# ZEDEDA Camera Monitor Agent - Summary

## What We Built

A complete computer monitor detection system that:

### 🎥 **Camera Monitoring**
- Captures frames from `/dev/video0` every 5 seconds (configurable)
- Uses OpenCV for camera access and image processing
- Saves detection images when monitors are found

### 🤖 **AI-Powered Detection**
- Integrates with Ollama LLaVA 7B vision model
- Analyzes camera frames to detect computer monitors, laptop screens, and displays
- Uses sophisticated prompts to identify various types of screens

### 📧 **Smart Alerting System**
- Configurable email alerts when monitors are detected
- Template-based email formatting with dynamic content
- Support for multiple alert rules and recipients
- Desktop notifications (optional)

### 🔧 **Flexible Configuration**
- YAML-based configuration file (`camera_config.yaml`)
- Environment variable support (`.env` file)
- Multiple detection rules and notification settings
- Adjustable confidence thresholds and keywords

### 🐳 **Multiple Deployment Options**

1. **Docker Compose** (Recommended)
   ```bash
   docker compose up -d
   ```

2. **Systemd Service**
   ```bash
   ./start_camera_agent.sh service
   sudo systemctl start zededa-camera-agent
   ```

3. **Direct Python Execution**
   ```bash
   python3 camera_agent.py --config camera_config.yaml
   ```

## File Structure

```
/home/nvidia/Developer/agent-attempt-2/
├── camera_agent.py              # Main camera monitoring agent
├── camera_config.yaml           # Configuration file
├── Dockerfile.camera            # Docker container for camera agent
├── docker-compose.yml           # Updated with camera agent service
├── zededa-camera-agent.service  # Systemd service file
├── start_camera_agent.sh        # Startup and management script
├── test_camera_setup.py         # System test script
├── demo_camera_agent.py         # Demo/example script
├── CAMERA_MONITOR_README.md     # Detailed documentation
└── detected_images/             # Directory for saved detection images
```

## Key Features

### ✅ **Working Components**
- Camera access to `/dev/video0` ✅
- Ollama integration with LLaVA 7B model ✅
- Configuration system ✅
- Docker containerization ✅
- Systemd service ✅
- Comprehensive testing ✅

### 📋 **Configuration Examples**

**Basic Monitor Detection:**
```yaml
rules:
  - id: "MONITOR_DETECTED"
    description: "Alert when computer monitor detected"
    enabled: true
    email:
      to: ["adithya7shankar@gmail.com"]
      subject: "🖥️ Computer Monitor Detected"
```

**Environment Variables:**
```bash
OLLAMA_URL=http://localhost:11434
VISION_MODEL=llava:7b
CAMERA_INDEX=0
CAPTURE_INTERVAL=5
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
```

## Usage Examples

### Quick Start
```bash
# Setup environment
./start_camera_agent.sh setup

# Test system
./start_camera_agent.sh check

# Start monitoring
./start_camera_agent.sh start
```

### Docker Deployment
```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f camera-agent

# Stop services
docker compose down
```

### Testing
```bash
# Quick test
python3 test_camera_setup.py

# Agent test mode
python3 camera_agent.py --test

# Demo mode
python3 demo_camera_agent.py
```

## How It Works

1. **Continuous Monitoring**: Camera captures frames every 5 seconds
2. **AI Analysis**: Each frame is sent to Ollama LLaVA for analysis
3. **Detection Logic**: AI response is checked for monitor-related keywords
4. **Rule Evaluation**: Detected monitors trigger configured alert rules
5. **Alert Delivery**: Email notifications sent to configured recipients
6. **Logging**: All activity logged with timestamps and statistics

## Current Status

### ✅ **Ready to Use**
- Camera monitoring agent is fully functional
- Configuration system is complete
- Docker deployment is working
- Ollama integration is operational
- Basic testing confirms all components work

### 🔧 **To Configure**
- Update email credentials in `.env` file
- Customize detection rules in `camera_config.yaml`
- Adjust capture interval and detection sensitivity
- Set up production monitoring and alerting

### 📈 **Future Enhancements**
- Web dashboard for monitoring status
- Webhook integration for external systems
- Multiple camera support
- Advanced detection algorithms
- Performance monitoring and metrics

## Security Considerations

- Camera agent runs as non-root user
- Device access limited to `/dev/video0`
- Email credentials stored in environment variables
- No external data transmission (except email alerts)
- Local image storage with configurable retention

## Integration with Existing System

The camera monitoring agent works alongside the existing ZEDEDA AI Agent infrastructure:
- Uses same Ollama instance for AI processing
- Shares Docker Compose configuration
- Compatible with existing email automation tools
- Can be deployed independently or as part of the full suite

This creates a comprehensive security monitoring solution that combines:
- Email automation (existing)
- Camera-based visual monitoring (new)
- AI-powered detection and alerting (both)