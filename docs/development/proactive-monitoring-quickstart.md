# Proactive Monitoring — Quick Start

Get the monitoring loop running and inspecting PCBs in three steps.

## Prerequisites

1. **VLM backend** running (vLLM or Ollama)
   ```bash
   vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
   ```

2. **Camera** available (or video file)
   ```bash
   ls /dev/video*
   ```

3. **Dependencies** installed
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 1 — Configure

Edit `config.yaml`:

```yaml
proactive:
  enabled: true
  frame_interval_seconds: 1.5       # How often to observe
  stationary_motion_threshold: 5.0   # Motion threshold for "stopped"

camera:
  device_index: 0

vllm:
  url: http://localhost:8000
  model: Qwen/Qwen3-VL-8B-Instruct
```

## Step 2 — Start the server

```bash
python run.py
# → http://localhost:8080
```

## Step 3 — Start monitoring

**Chat UI:**
1. Navigate to `http://localhost:8080`
2. Type: *"Monitor the conveyor for PCB defects"*
3. The agent starts `MonitoringLoop`, observes the feed, and runs VLM
   inspection whenever a board stops in the zone.

**API:**
```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Monitor the conveyor for PCB defects"}'
```

---

## What happens under the hood

1. `MonitoringLoop` subscribes to the camera feed.
2. Per frame, it computes motion score, board-in-zone, and board signature
   using OpenCV.
3. When a board **stops** in the zone (boolean edge detection), it fires
   `on_board_ready(frame, context)`.
4. The service callback calls the VLM for defect analysis.
5. The result is persisted to `pcb_inspections` and emitted to the chat UI.
6. The LLM decides follow-up actions (alert, log, etc.) via MCP tools.

---

## Checking status

```bash
curl http://localhost:8080/api/monitoring/proactive/status
```

```json
{
  "success": true,
  "status": {
    "running": true,
    "context": {
      "instruction": "Monitor the conveyor for PCB defects",
      "frames_processed": 145,
      "inspections_completed": 4,
      "defect_found_count": 1,
      "no_defect_count": 3
    }
  }
}
```

## Stopping

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/stop
```

Or in the chat UI: *"Stop monitoring"*.

---

## Example scenario — PCB defect alerting

```
User: "Monitor for defective PCBs and alert admin@acme.com"

MonitoringLoop observes camera feed …
  Board enters zone → motion score drops → board stops
  on_board_ready fires → VLM analysis → FAIL (solder bridge)
  LLM calls send_defect_alert(recipients=["admin@acme.com"])

Agent: "🚨 Defect detected — solder bridge on U3. Alert sent to admin@acme.com"
```

---

## Configuration reference

| Key | Default | Effect |
|-----|---------|--------|
| `proactive.enabled` | `true` | Enable/disable proactive mode |
| `proactive.frame_interval_seconds` | `1.5` | Seconds between CV observations |
| `proactive.fast_observation_mode` | `true` | Use fast CV path |
| `proactive.stationary_motion_threshold` | `5.0` | Motion score below = "stopped" |

Zone crop, segmentation, and tracking parameters are also available in
`config/defaults.yaml`.

---

## Troubleshooting

### Agent not starting

1. Is VLM running? `curl http://localhost:8000/v1/models`
2. Is camera available? Check `GET /api/status`
3. Check logs: `tail -f agent.log`

### No inspections happening

1. Is a board actually stopping in frame?
2. Check `GET /api/monitoring/proactive/status` →
   `last_observation` should show `board_in_zone: true`
3. Lower `stationary_motion_threshold` if boards register as "moving"

### Too many inspections

1. Increase `frame_interval_seconds`
2. Raise `stationary_motion_threshold`

---

## Next steps

- [Proactive monitoring architecture](../architecture/README.md)
- [API endpoints](../api/endpoints.md)
- [Agent core README](../agent/README.md)
