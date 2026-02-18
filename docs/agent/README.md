# Core Agent Modules

This directory contains the core agent runtime for the PCB conveyor inspection system.

## Architecture Principle

**The monitoring loop is the ONLY deterministic piece.** Everything else — inspection, alerting, classification, logging — is decided by the LLM through MCP tool calls.

```
Camera Feed
    │
    ▼
MonitoringLoop._observe()       ← CV sensor readings (deterministic)
    │
    ▼
State-change detected?
    │ YES
    ▼
on_board_ready(frame, context)  ← LLM decides which tools to call
                                   • inspect_pcb / inspect_pcb_frame
                                   • send_defect_alert
                                   • log_defect
                                   • classify_board
                                   • (or: do nothing)
```

## Modules

### `monitoring_loop.py` — Monitoring Loop (deterministic observation)

**Purpose:** Frame acquisition + CV sensor readings + state-change detection. This is the *only* deterministic component in the agent system.

**What it does:**
- ✅ Acquires frames from the camera publisher
- ✅ Computes CV sensor readings (motion, edge density, board presence)
- ✅ Tracks board identity via contour tracking + perceptual hash
- ✅ Detects state changes (board stopped in inspection zone)
- ✅ Fires `on_board_ready` callback when a board is ready
- ✅ Stores frames to disk/DB when PCB detected + motion low (mechanical)
- ✅ PCB-only scope enforcement (non-PCB instructions are refused)

**What it does NOT do:**
- ❌ Decide whether to inspect
- ❌ Decide what to inspect for
- ❌ Classify defects
- ❌ Send alerts
- ❌ Score frame quality for gating

**Classes:**

#### `MonitoringLoop`

```python
from agents.core.monitoring_loop import MonitoringLoop

loop = MonitoringLoop(
    instruction="Monitor the conveyor for PCB defects",
    publisher_getter=get_camera_publisher,
    on_board_ready=handle_board_ready,  # LLM decides here
    event_callback=record_event,
    config={
        "frame_interval_seconds": 0.1,
        "stationary_motion_threshold": 1.8,
    },
)

loop.start()
```

**Configuration (CV sensor tuning — not decisions):**
- `frame_interval_seconds`: Frame acquisition pacing
- `stationary_motion_threshold`: Motion score below which board is "stopped"
- `zone_crop_top_ratio` / `zone_crop_bottom_ratio`: ROI for board detection
- `zone_presence_threshold`: Edge density threshold for board presence
- `presence_confirm_frames` / `absence_confirm_frames`: Hysteresis depths
- `track_iou_threshold`: IoU threshold for board identity tracking
- `segmentation_min_area_ratio` / `segmentation_max_area_ratio`: Contour filters

#### `Observation` DataClass

CV sensor readings from a single frame — **no decisions**:

```python
@dataclass
class Observation:
    frame_number: int
    timestamp: str
    motion_score: float       # Raw motion from ROI diff
    motion_moving: bool       # motion_score > threshold
    edge_density: float       # Edge pixel percentage in ROI
    board_in_zone: bool       # Debounced board presence
    board_signature: str      # track-N or board-<hash>
    frame_quality: dict       # sharpness, brightness, contrast
```

#### `MonitoringContext` DataClass

Runtime metrics provided to the LLM as context:

```python
@dataclass
class MonitoringContext:
    instruction: str
    frames_processed: int
    inspections_completed: int
    defect_found_count: int
    no_defect_count: int
    last_observation: Optional[Observation]
    last_action_time: float
    board_decisions: Dict[str, str]  # signature → DEFECT_FOUND/NO_DEFECT
```

**Methods:**
- `start()` — Start the monitoring loop
- `stop()` — Stop the monitoring loop
- `update_instruction(instruction)` — Update PCB inspection objective
- `snapshot()` — Runtime state for dashboards and LLM context
- `get_performance_metrics()` — Frames processed, inspections completed, etc.

### `detection_agent.py` — VLM Analysis Coordinator

**Purpose:** Executes VLM-based vision analysis tasks when the LLM decides to inspect.

**Key Classes:**
- `StreamlinedAgent` — Unified VLM client wrapper with SSIM skip, circuit breaker, and memory

**Key Methods:**
- `analyze_frame(frame)` — Standard VLM analysis
- `analyze_with_prompt(frame, task_type, custom_prompt)` — Task-specific analysis
- `analyze_agentic(frame, task_type, custom_prompt, recipients)` — Analysis with tool calling

The `StreamlinedAgent` does NOT decide when to analyze. It's called by the LLM via MCP tools or by the `on_board_ready` callback.

### `state.py` — Data Models

**Purpose:** Shared data structures used across the system.

**Key Classes:**
- `DetectionEvent` — Represents a PCB inspection event (timestamp, confidence, labels, traces)
- `AgentMemory` — Thread-safe ring buffer for event history
- `AgentSnapshot` — UI-facing state snapshot

## Deleted Modules (formerly in this directory)

| Module | Reason for removal |
|--------|-------------------|
| `conveyor_inspection_fsm.py` | Hardcoded FSM replaced by LLM-driven decisions |
| `defect_monitor.py` | Duplicate polling loop — LLM uses existing MCP tools |
| `alerting.py` | Hardcoded alert dispatch — LLM calls alert tools (`send_defect_alert` / `send_alert_email`) |
| `opencv_pcb_presence_test.py` | Dead code (zero importers) |

## Backward-Compatibility Modules (still present)

These files remain for import stability but are wrappers/re-exports:

| Module | Current behavior |
|--------|------------------|
| `proactive_monitoring.py` | Deprecated shim that re-exports `MonitoringLoop` from `monitoring_loop.py` |
| `resilience.py` | Re-exports `CircuitBreaker`/`CircuitState` from `core/resilience.py` |

## Integration Example

```python
from services.core.monitoring import StreamlinedMonitoringService

service = StreamlinedMonitoringService()
service.initialize()

# Start monitoring — the loop observes, the LLM decides
service.start_monitoring(
    instruction="Inspect PCBs for solder bridges and lifted pads",
)

# Check status
status = service.get_proactive_snapshot()
print(f"Running: {status['running']}")
print(f"Frames: {status['context']['frames_processed']}")

# Stop
service.stop_monitoring()
```

## Testing

```bash
pytest tests/unit/agents/ -v
pytest tests/integration/test_proactive_monitoring.py -v
```

## Key Principles

1. **Observation ≠ Decision.** The loop observes; the LLM decides.
2. **Sensor readings, not rules.** CV code answers "what is happening?" — never "what should we do?"
3. **One deterministic piece.** Only `MonitoringLoop` is hardcoded flow. Everything else is LLM tool calls.
4. **Clear separation.** Data models in `state.py`, VLM calls in `detection_agent.py`, loop in `monitoring_loop.py`.
