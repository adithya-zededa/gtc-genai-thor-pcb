# Proactive Monitoring Quick Start Guide

Get up and running with the intelligent, LLM-driven proactive monitoring system in minutes.

## Prerequisites

1. **Running VLM Backend** (vLLM or Ollama)
   ```bash
   # Example: vLLM with Qwen3-VL
   vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
   ```

2. **Camera Available** (or video file for testing)
   ```bash
   # Check camera availability
   ls /dev/video*
   ```

3. **Dependencies Installed**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start: 3 Steps

### Step 1: Configure Proactive Mode

Edit `config.yaml`:

```yaml
proactive:
  enabled: true  # Enable proactive mode
  frame_interval_seconds: 1.5
  decision_temperature: 0.2

camera:
  device_index: 0  # Your camera index
  
vllm:
  url: http://localhost:8000
  model: Qwen/Qwen3-VL-8B-Instruct
```

### Step 2: Start the Server

```bash
python run.py
```

The server starts at `http://localhost:8080`

### Step 3: Start Proactive Monitoring

**Via UI:**
1. Navigate to `http://localhost:8080`
2. Go to monitoring dashboard
3. Click "Start Proactive Monitoring"
4. Enter instruction: `"Monitor the conveyor for PCB defects"`

**Via API:**
```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Monitor the conveyor for PCB defects"
  }'
```

**Response:**
```json
{
  "success": true,
  "status": {
    "running": true,
    "instruction": "Monitor the conveyor for PCB defects",
    "context": {
      "frames_processed": 0
    }
  }
}
```

## Monitoring Status

### Check Agent Status

```bash
curl http://localhost:8080/api/monitoring/proactive/status
```

**Response:**
```json
{
  "success": true,
  "status": {
    "running": true,
    "instruction": "Monitor the conveyor for PCB defects",
    "context": {
      "frames_processed": 145,
      "frames_since_scene_change": 8,
      "last_action": "wait",
      "quick_check_count": 5,
      "full_inspection_count": 2,
      "last_observation": {
        "scene_summary": "Empty conveyor belt visible",
        "target_present": false,
        "target_state": "gone"
      },
      "last_decision": {
        "action": "wait",
        "confidence": 0.95,
        "reasoning": "Scene is empty, no PCB present, continue monitoring"
      }
    }
  }
}
```

### Check System Status

```bash
curl http://localhost:8080/api/status
```

## Example Scenarios

### Scenario 1: PCB Inspection

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Monitor the conveyor for PCB defects. Inspect each board when it stops moving.",
    "frame_interval_seconds": 1.0,
    "stability_frame_count": 5
  }'
```

**What happens:**
1. Agent waits while conveyor is empty
2. Detects PCB entering frame
3. Observes PCB moving into position
4. Quick-checks PCB state
5. Waits for PCB to stabilize
6. Runs full PCB inspection when stable
7. Avoids re-inspecting same PCB

### Scenario 2: PPE Compliance

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Check that all workers are wearing hard hats and reflective vests.",
    "frame_interval_seconds": 2.0
  }'
```

**What happens:**
1. Agent waits while area is empty
2. Detects person entering frame
3. Quick-checks person count
4. Runs PPE compliance check when scene stable
5. Re-checks if new people arrive
6. Emits alerts for non-compliance

### Scenario 3: Custom Monitoring

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Watch for any objects left unattended for more than 30 seconds.",
    "frame_interval_seconds": 1.5
  }'
```

**What happens:**
1. LLM interprets custom objective
2. Tracks objects over time
3. Uses temporal context to measure duration
4. Decides when threshold is met
5. Generates alerts based on instruction

## Viewing Results

### Real-time WebSocket Updates

Connect to WebSocket for live events:
```javascript
const socket = io('http://localhost:8080');
socket.on('new_log', (data) => {
  console.log('Detection event:', data);
});
```

### Check Detection Logs

```bash
curl http://localhost:8080/api/logs?limit=10
```

### View Detection Images

Detection images are saved to `detected_images/` directory by default.

## Stopping Monitoring

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/stop
```

## Configuration Reference

### Frame Processing

```yaml
proactive:
  frame_interval_seconds: 1.5  # How often to process frames
  stability_frame_count: 6  # Frames needed for "stable" hint
```

**Tuning:**
- Faster scenes → lower frame_interval (1.0-1.5s)
- Slower scenes → higher frame_interval (2.0-3.0s)
- More stability assurance → higher stability_frame_count

### LLM Temperature

```yaml
proactive:
  observation_temperature: 0.1  # Consistent observations
  decision_temperature: 0.2  # Balanced decisions
  quick_check_temperature: 0.15  # Fast confirmations
```

**Tuning:**
- More deterministic → lower (0.05-0.1)
- More exploratory → higher (0.2-0.3)
- Keep observation_temperature low for consistency

### Context Management

```yaml
proactive:
  inspection_ttl_seconds: 180  # Scene memory duration
  max_idle_seconds: 300  # Idle time awareness
```

**Tuning:**
- Frequently changing scenes → lower ttl (60-120s)
- Static scenes → higher ttl (300-600s)

## Troubleshooting

### Agent Not Starting

**Issue:** `start_proactive_monitoring` returns error

**Check:**
1. Is VLM backend running?
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. Is camera available?
   ```bash
   curl http://localhost:8080/api/status
   # Check "camera_available": true
   ```

3. Check logs:
   ```bash
   tail -f agent.log
   ```

### No Actions Being Taken

**Issue:** Agent always decides "wait"

**Check:**
1. Is the instruction clear and specific?
2. Is the target object actually visible in frame?
3. Check observation results in status:
   ```bash
   curl http://localhost:8080/api/monitoring/proactive/status
   # Look at last_observation.target_present
   ```

4. Increase decision_temperature for more exploratory behavior:
   ```yaml
   decision_temperature: 0.3  # More willing to take action
   ```

### Too Many Actions

**Issue:** Agent running too many inspections

**Check:**
1. Is frame_interval_seconds too low?
2. Is scene signature maintained consistently?
3. Check inspection_ttl_seconds:
   ```yaml
   inspection_ttl_seconds: 300  # Remember longer
   ```

### Poor Decision Quality

**Issue:** Agent making suboptimal decisions

**Check:**
1. Review decision reasoning in logs
2. Verify instruction is clear and specific
3. Check observation quality:
   - Is confidence high?
   - Are target_state values accurate?
4. Adjust temperatures:
   ```yaml
   observation_temperature: 0.05  # More deterministic observations
   decision_temperature: 0.25  # More reasoning depth
   ```

## Performance Tips

### Optimize for Speed

```yaml
proactive:
  frame_interval_seconds: 2.0  # Process fewer frames
  observation_temperature: 0.05  # Faster, more deterministic
```

### Optimize for Accuracy

```yaml
proactive:
  frame_interval_seconds: 1.0  # Process more frames
  stability_frame_count: 8  # Wait longer for stability
  decision_temperature: 0.25  # More thoughtful decisions
```

### Optimize for Efficiency

```yaml
proactive:
  frame_interval_seconds: 1.5  # Balanced
  inspection_ttl_seconds: 300  # Long memory
  decision_temperature: 0.2  # Balanced reasoning
```

## Next Steps

- Read [Proactive Monitoring Architecture](../architecture/proactive-monitoring.md)
- Explore [API Documentation](../api/endpoints.md)
- Review example [monitoring scenarios](../examples/monitoring-scenarios.md)
- Learn about [prompt engineering](./prompt-engineering.md) for better instructions

## Support

For issues or questions:
- Check logs: `tail -f agent.log`
- Review status endpoint: `GET /api/monitoring/proactive/status`
- Review system status: `GET /api/status`
- Enable debug logging in `config.yaml`
