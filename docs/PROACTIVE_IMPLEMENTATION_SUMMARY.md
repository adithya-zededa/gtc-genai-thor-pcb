# Proactive Monitoring — Implementation Summary

## Overview

The camera monitoring system uses a **single deterministic observation loop**
(`MonitoringLoop`) that delegates every decision to the LLM via MCP tools.

---

## Architecture

```
Camera Feed  →  MonitoringLoop (CV only)
                     │
                     │  on_board_ready(frame, context)
                     ▼
              Service callback
                ├── VLM analysis → DetectionEvent
                ├── Persist to pcb_inspections
                ├── Emit SocketIO event → Chat UI
                └── LLM decides follow-up via MCP tools
```

### What is deterministic

Only the observation loop:

- **Motion scoring** — ROI diff between frames
- **Board-in-zone detection** — contour segmentation + hysteresis
- **Board signature tracking** — tracker ID or content hash
- **State-change detection** — `board just stopped?` boolean edge

### What the LLM decides

Everything else:

| Decision | Mechanism |
|----------|-----------|
| Whether to inspect | `on_board_ready` callback fires VLM |
| What prompt to use | Active task prompt from session |
| Severity classification | VLM response → DetectionEvent |
| Whether to alert | `send_defect_alert` MCP tool |
| What to log/report | LLM chains tools as needed |
| When to stop | User says "stop" → `end_session` MCP tool |

---

## Key Components

### 1. MonitoringLoop
**File:** `agents/core/monitoring_loop.py` (~660 lines)

- Subscribes to camera feed
- Computes `Observation` dataclass per frame (sensor readings only)
- Fires `on_board_ready(frame, MonitoringContext)` when a board stops
- Maintains counters and context across frames
- Does **not** call VLM, classify defects, or send alerts

### 2. StreamlinedAgent (VLM)
**File:** `agents/core/detection_agent.py` (~567 lines)

- Called by the service callback when `on_board_ready` fires
- Runs VLM analysis with the session's active prompt
- Returns `DetectionEvent` with classification & description

### 3. State Models
**File:** `agents/core/state.py` (~194 lines)

- `DetectionEvent` — single inspection result
- `AgentMemory` — event history
- `AgentSnapshot` — serialisable state

### 4. Service Integration
**File:** `services/core/monitoring.py`

- `_on_board_ready(frame, context)` — wires VLM + persistence
- `_persist_inspection(event, board_signature)` — database write
- Emits SocketIO events for the chat UI

---

## What Was Removed

| Old component | Lines | Replacement |
|---|---|---|
| `proactive_monitoring.py` (god class) | ~1133 | `MonitoringLoop` (~660) |
| `conveyor_inspection_fsm.py` (4-state FSM) | ~150 | Boolean edge detection |
| `defect_monitor.py` (polling consumer loop) | ~300 | `on_board_ready` callback |
| `alerting.py` (duplicate of email tool) | ~180 | `send_defect_alert` MCP tool |
| `opencv_pcb_presence_test.py` (dead code) | ~90 | Deleted |

**Net reduction:** ~3100 lines consolidated into ~660 lines.

Backward-compatibility shims exist:
- `agents/core/proactive_monitoring.py` → re-exports `MonitoringLoop` as `ProactiveMonitoringAgent`
- `agents/core/resilience.py` → re-exports from `core.resilience`

---

## Configuration

Only CV-related settings are consumed by `MonitoringLoop`:

```yaml
proactive:
  enabled: true
  frame_interval_seconds: 1.5
  fast_observation_mode: true
  stationary_motion_threshold: 5.0
```

Temperature, TTL, and idle-timeout settings exist in config but are **not**
consumed by the loop.

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/monitoring/proactive/start` | POST | Start with instruction |
| `/api/monitoring/proactive/stop` | POST | Stop the loop |
| `/api/monitoring/proactive/status` | GET | Status + context snapshot |

---

## Usage

### Via Chat UI

```
User: "Monitor the conveyor for PCB defects and alert admin@acme.com"
Agent: starts MonitoringLoop → observes → board stops →
       VLM inspection → defect found →
       calls send_defect_alert → "🚨 Defect detected, alert sent."
```

### Via API

```bash
curl -X POST http://localhost:8080/api/monitoring/proactive/start \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Monitor the conveyor for PCB defects"}'
```

---

## Source Files

- `agents/core/monitoring_loop.py` — observation loop (deterministic)
- `agents/core/detection_agent.py` — VLM analysis agent
- `agents/core/state.py` — DetectionEvent, AgentMemory
- `services/core/monitoring.py` — callback + persistence
- `app/api/v1/monitoring.py` — REST endpoints
- `config/defaults.yaml` — default configuration

---

**Status:** Production-ready
