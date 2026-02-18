# Proactive Monitoring Runtime (Current Implementation)

## Overview

This document describes the **current** proactive monitoring runtime.

The runtime is a **single deterministic observation loop** (`MonitoringLoop`)
that feeds every decision to the LLM via an `on_board_ready` callback.
The only deterministic code is *sensing* — motion scoring, board-in-zone
detection, and board signature tracking — using OpenCV.  Everything else
(inspection trigger, severity classification, alerting) is decided by the
LLM through MCP tool calls.

## What Is Running Today

- `MonitoringLoop` subscribes to the camera feed and runs CV observation.
- When a board stops in the inspection zone, the loop fires
  `on_board_ready(frame, context)`.
- The service layer (`StreamlinedMonitoringService._on_board_ready`) runs
  VLM analysis and persists the result.  The LLM decides follow-up
  actions (alert, log, report) via MCP tools.
- There is **no** separate defect-monitoring consumer loop, **no**
  hardcoded FSM, and **no** deterministic frame-quality selector.

## High-Level Flow

```
User instruction
  → MonitoringLoop.start()
     → Subscribe to camera feed
     → Per-frame CV observation
        ├── Motion score (ROI diff)
        ├── Board-in-zone (contour + hysteresis)
        └── Board signature (tracker or hash)
     → State-change detection
        just_stopped = board_stopped AND NOT prev_board_stopped
     → If just_stopped:
          on_board_ready(frame, MonitoringContext)
            → VLM inspection (StreamlinedAgent)
            → Persist inspection (PASS / FAIL)
            → Emit SocketIO event → Chat UI
            → LLM decides next actions via MCP tools
```

## Runtime Components

### 1) MonitoringLoop  (`agents/core/monitoring_loop.py`)

Single class (~660 lines) that owns:

| Responsibility | What it does |
|---|---|
| **Frame pacing** | Respects `frame_interval_seconds` between observations |
| **CV observation** | Computes `Observation` dataclass per frame |
| **State-change detection** | Detects board-just-stopped via boolean edge |
| **Callback dispatch** | Fires `on_board_ready(frame, context)` once per board stop |
| **Context bookkeeping** | Maintains `MonitoringContext` (counters, last observation, etc.) |

The loop does **not** call the VLM, classify defects, or send alerts.

### 2) Observation (dataclass)

Pure sensor readings — no decisions:

```python
@dataclass
class Observation:
    motion_score: float
    board_in_zone: bool
    board_signature: str | None
    edge_density: float
    raw_contours: int
    timestamp: float
```

### 3) MonitoringContext (dataclass)

Running state carried across frames:

```python
@dataclass
class MonitoringContext:
    instruction: str
    frames_processed: int
    inspections_started: int
    inspections_completed: int
    defect_found_count: int
    no_defect_count: int
    last_observation: Observation | None
    last_action_time: float
    last_decision: str | None
    last_decision_reason: str | None
    board_decisions: dict[str, str]
```

### 4) on_board_ready callback  (`services/core/monitoring.py`)

Wired up by `StreamlinedMonitoringService._on_board_ready`:

1. Calls `StreamlinedAgent.analyze_with_prompt()` (VLM) on the frame.
2. Persists the inspection result to `pcb_inspections`.
3. Emits a SocketIO event to the chat UI.
4. Updates `MonitoringContext` counters.
5. The LLM decides whether to call `send_defect_alert`, `log_defect`,
   or any other MCP tool.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/monitoring/proactive/start` | POST | Start monitoring with instruction + config |
| `/api/monitoring/proactive/stop` | POST | Stop the loop |
| `/api/monitoring/proactive/status` | GET | Loop status + context snapshot |

## Configuration (actively used)

| Key | Default | Used by |
|---|---|---|
| `frame_interval_seconds` | 1.5 | MonitoringLoop frame pacing |
| `fast_observation_mode` | true | CV observation method |
| `stationary_motion_threshold` | 5.0 | Board-stopped detection |
| Zone crop / presence controls | — | Board-in-zone segmentation |
| Segmentation / tracking controls | — | Board signature tracking |

### Not used by the monitoring loop

These config keys exist but are not consumed by MonitoringLoop:

- `decision_temperature`, `quick_check_temperature` — no LLM call inside the loop
- `observation_temperature` — observation is CV-only
- `inspection_ttl_seconds`, `max_idle_seconds` — no idle/TTL logic in the loop
- `stability_frame_count` — no frame-quality selector

## Deleted Components

| Old file | What it did | Replacement |
|---|---|---|
| `conveyor_inspection_fsm.py` | 4-state hardcoded FSM | `MonitoringLoop` boolean edge detection |
| `defect_monitor.py` | Background polling consumer loop | `on_board_ready` callback |
| `proactive_monitoring.py` | 1133-line god class | `MonitoringLoop` (~660 lines) |
| `alerting.py` | Duplicate of `agents/tools/email.py` | `send_defect_alert` MCP tool |
| `opencv_pcb_presence_test.py` | Dead code (zero importers) | Deleted |

## Source of Truth

Primary implementation files:

- `agents/core/monitoring_loop.py` — the only deterministic piece
- `agents/core/detection_agent.py` — VLM analysis (called by callback)
- `services/core/monitoring.py` — wires callback + persistence
- `app/api/v1/monitoring.py` — REST endpoints
- `config/defaults.yaml` — default configuration

---
