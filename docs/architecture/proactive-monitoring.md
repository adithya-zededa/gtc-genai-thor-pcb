# Proactive Monitoring Architecture

## Overview

The proactive monitoring system watches a camera feed for PCB boards on a
conveyor and delegates **every decision** to the LLM.  The only deterministic
code is a lightweight OpenCV observation loop (`MonitoringLoop`) that detects
when a board has stopped in the inspection zone.

## Core Principle

> **Except for the monitoring loop, every tool call or action is decided by
> the LLM.**

The system is split into two halves:

| Half | What it does | Where |
|------|-------------|-------|
| **Deterministic** | CV observation — motion, board-in-zone, signature | `agents/core/monitoring_loop.py` |
| **LLM-decided** | Inspection, classification, alerting, logging | MCP tools + `on_board_ready` callback |

## Architecture Diagram

```
┌──────────────────────────────────────────────────┐
│           User: "Monitor for PCB defects"        │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│               MonitoringLoop (CV)                │
│                                                  │
│   Per-frame:                                     │
│     motion_score  = ROI diff                     │
│     board_in_zone = contour + hysteresis         │
│     board_sig     = tracker or hash              │
│                                                  │
│   State-change:                                  │
│     just_stopped = stopped AND NOT prev_stopped  │
└────────────────────────┬─────────────────────────┘
                         │ on_board_ready(frame, ctx)
                         ▼
┌──────────────────────────────────────────────────┐
│          Service layer (_on_board_ready)          │
│                                                  │
│   1. VLM analysis  →  DetectionEvent             │
│   2. Persist to pcb_inspections (PASS / FAIL)    │
│   3. Emit SocketIO event → Chat UI               │
│   4. Update MonitoringContext counters            │
│   5. LLM decides follow-up actions via MCP:      │
│        send_defect_alert, log_defect, …          │
└──────────────────────────────────────────────────┘
```

## MonitoringLoop Details

`MonitoringLoop` (~660 lines) is the **only** deterministic component:

- Subscribes to camera feed via `CameraService`
- Runs `_observe(frame)` → `Observation` dataclass (sensor readings only)
- Detects state change: `board just stopped in zone?`
- Fires `on_board_ready(frame, MonitoringContext)` **once** per board stop
- Maintains `MonitoringContext` counters (frames processed, inspections, etc.)

It does **not**:
- Call the VLM
- Classify defects
- Send alerts
- Make any quality/severity judgements

### Observation Dataclass

```python
@dataclass
class Observation:
    motion_score: float        # 0.0 (still) – 100.0 (moving fast)
    board_in_zone: bool        # Contour-based board detection
    board_signature: str | None  # Tracker ID or hash
    edge_density: float        # Edge pixel ratio
    raw_contours: int          # Number of contours found
    timestamp: float
```

### MonitoringContext Dataclass

```python
@dataclass
class MonitoringContext:
    instruction: str           # User's natural-language objective
    frames_processed: int
    inspections_started: int
    inspections_completed: int
    defect_found_count: int
    no_defect_count: int
    last_observation: Observation | None
    last_action_time: float
    last_decision: str | None
    last_decision_reason: str | None
    board_decisions: dict[str, str]  # signature → "PASS"/"FAIL"
```

## What the LLM Decides

Everything after observation is LLM-controlled:

| Decision | How |
|----------|-----|
| **Whether to inspect** | `on_board_ready` fires; service calls VLM |
| **What prompt to use** | Active task prompt from the session |
| **Severity classification** | VLM response parsed into DetectionEvent |
| **Whether to alert** | LLM calls `send_defect_alert` MCP tool |
| **What to log** | LLM calls `log_defect` or other tools |
| **When to stop** | User says "stop" → `end_session` MCP tool |

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/monitoring/proactive/start` | POST | Start with instruction + config |
| `/api/monitoring/proactive/stop` | POST | Stop the loop |
| `/api/monitoring/proactive/status` | GET | Loop status + context snapshot |

### Start Example

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Monitor the conveyor for PCB defects"}'
```

### Status Example

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
      "frames_processed": 312,
      "inspections_completed": 8,
      "defect_found_count": 1,
      "no_defect_count": 7,
      "last_decision": "PASS"
    }
  }
}
```

## Configuration

Only CV-related settings affect `MonitoringLoop` directly:

```yaml
proactive:
  enabled: true
  frame_interval_seconds: 1.5
  fast_observation_mode: true
  stationary_motion_threshold: 5.0
  # Zone crop and segmentation controls …
```

Temperature, TTL, and idle settings exist in config but are **not** consumed
by the loop — they are available for future LLM-side tuning if needed.

## Comparison: Old vs Current

| Aspect | Old (v1) | Current |
|--------|----------|---------|
| **Core control** | 4-state hardcoded FSM + DefectMonitorLoop | MonitoringLoop + on_board_ready callback |
| **Inspection trigger** | FSM state `INSPECTING` | Board-just-stopped boolean edge |
| **Frame quality gating** | Deterministic stability streak + quality threshold | Reported as metrics, LLM decides |
| **Alert dispatch** | Hardcoded AlertManager (duplicate of email tool) | `send_defect_alert` MCP tool |
| **Severity classification** | Hardcoded prompt inside defect_monitor.py | LLM classifies via VLM |
| **Files** | 5 files, ~3100 lines | 1 file, ~660 lines |

## Source Files

- `agents/core/monitoring_loop.py` — observation loop
- `agents/core/detection_agent.py` — VLM analysis agent
- `agents/core/state.py` — DetectionEvent, AgentMemory
- `services/core/monitoring.py` — callback wiring + persistence
- `app/api/v1/monitoring.py` — REST endpoints

---
