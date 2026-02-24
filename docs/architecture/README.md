# Architecture — PCB Conveyor Inspection Agent

## System Overview

The PCB Conveyor Inspection Agent is an **AI-powered industrial monitoring system** that watches a conveyor belt via camera, detects PCBs, inspects them for defects using a Vision Language Model (VLM), and alerts operators by email. Every decision flows through the LLM agent via explicit MCP tool calls — there are no hardcoded decision shortcuts.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         User (Chat UI)                               │
│   "Start monitoring for defective PCBs and email me@acme.com"        │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ SocketIO
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Flask App + WebSocket                           │
│   app/__init__.py · app/websocket/chat.py                            │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        MCP Manager                                   │
│   agents/mcp/manager.py                                              │
│   ┌──────────────┐    ┌──────────────────────────────────────┐       │
│   │  LLM Intent  │───▶│  Domain Router                       │       │
│   │  Classifier  │    │  ┌────────────┐  ┌────────────────┐  │       │
│   └──────────────┘    │  │  general    │  │  pcb           │  │       │
│                       │  │  domain     │  │  domain        │  │       │
│                       │  └────────────┘  └────────────────┘  │       │
│                       └──────────────────────────────────────┘       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌───────────┐ ┌──────────────┐
     │  Tool        │ │  VLM      │ │  Monitoring   │
     │  Handlers    │ │  Client   │ │  Service      │
     │  agents/     │ │  agents/  │ │  services/    │
     │  tools/      │ │  vlm/     │ │  core/        │
     └──────────────┘ └───────────┘ └──────────────┘
              │              │              │
              ▼              ▼              ▼
     ┌──────────────┐ ┌───────────┐ ┌──────────────┐
     │  Email       │ │  vLLM     │ │  Camera Feed  │
     │  Service     │ │  Backend  │ │  Publisher    │
     └──────────────┘ └───────────┘ └──────────────┘
```

---

## Core Design Principles

1. **Language is the only control surface.** Users interact via natural language chat. The LLM agent decides what tools to call.
2. **Propose → Approve → Execute.** All tool calls follow an explicit lifecycle (`MCPToolCallProposal`). Sensitive actions require user confirmation.
3. **Frame capture is decoupled from the LLM.** The CV pipeline stores frames automatically; the LLM accesses them via tool calls, never touching the camera directly.
4. **One responsibility per tool.** Retrieving frames, inspecting frames, sending alerts, and querying history are all separate tools the LLM chains together.

---

## Directory Structure

```
├── agents/                    # Agent logic
│   ├── classifiers/           # LLM-based intent classifier
│   │   └── llm_classifier.py  # Routes messages → domain + tool
│   ├── core/                  # Core agent runtime
│   │   ├── monitoring_loop.py # Observation loop (only deterministic piece)
│   │   ├── detection_agent.py # Streamlined VLM detection agent
│   │   └── state.py           # DetectionEvent model
│   ├── mcp/                   # Model Context Protocol
│   │   ├── manager.py         # Domain router (pcb/general)
│   │   ├── executor_base.py   # Base executor with proposal lifecycle
│   │   ├── executor.py        # General domain executor
│   │   ├── interpreter.py     # General domain interpreter
│   │   ├── lifecycle.py       # MCPToolCallProposal + MCPToolResult
│   │   ├── state_machine.py   # AgentStateMachine (6 states)
│   │   ├── registry.py        # MCPToolDefinition + MCPToolRegistry
│   │   └── domains/
│   │       └── pcb.py         # PCB domain: registry, interpreter, executor
│   ├── tools/                 # Tool handler functions
│   │   ├── pcb.py             # PCB tools (inspect, alert, monitor, query)
│   │   ├── email.py           # Email sending
│   │   └── base.py            # General tools (history, evidence, etc.)
│   └── vlm/                   # Vision Language Model
│       ├── client.py          # UnifiedVLMClient (vLLM / Ollama)
│       ├── prompts.py         # Task-specific VLM prompts
│       └── task_types.py      # TaskType enum
├── app/                       # Flask application
│   ├── __init__.py            # App factory (create_app)
│   ├── api/v1/                # REST API endpoints
│   ├── database/              # SQLite models + repositories
│   │   ├── connection.py      # DB init, connection pool
│   │   ├── models.py          # Dataclass models (8 models)
│   │   └── repositories.py   # Repository classes (9 repos)
│   ├── views/                 # Jinja2 HTML views
│   └── websocket/
│       └── chat.py            # SocketIO chat handlers
├── config/                    # Pydantic settings + YAML defaults
├── core/                      # Cross-cutting concerns
│   ├── config.py              # Configuration loading
│   ├── logging.py             # Structured logging
│   └── errors.py              # Error hierarchy (CameraAgentError)
├── router/                    # LLM Router
│   ├── llm_router.py          # vLLM routing + token tracking
│   └── adapters/              # Provider adapters
├── services/                  # Business logic services
│   ├── core/
│   │   ├── monitoring.py      # StreamlinedMonitoringService
│   │   ├── camera.py          # Camera feed publisher
│   │   └── pcb_presence_cv.py # OpenCV board detection
│   └── domains/pcb/
│       └── service.py         # PCB domain service (defect recording)
├── helm/                      # Kubernetes Helm chart
├── templates/                 # Jinja2 HTML templates
└── static/                    # CSS / JS
```

---

## Agent State Machine

The agent runs through 6 states with validated transitions:

```
         ┌─────────┐
         │   OFF   │──────────────────┐
         └────┬────┘                  │
              │ start                 │ shutdown
              ▼                       │
         ┌─────────┐                  │
    ┌────│  IDLE   │◀─────────────────┤
    │    └────┬────┘                  │
    │         │ start monitoring      │
    │         ▼                       │
    │    ┌────────────┐     ┌─────────┴──┐
    │    │ MONITORING │────▶│  ANALYZING  │
    │    └─────┬──────┘     └─────┬──────┘
    │          │                  │
    │          ▼                  ▼
    │    ┌──────────┐       ┌─────────┐
    │    │ ALERTING │       │  ERROR  │
    │    └──────────┘       └─────────┘
    │          │                  │
    └──────────┴──────────────────┘
                  → IDLE
```

| State | Description |
|-------|-------------|
| `OFF` | Agent powered down |
| `IDLE` | Ready, no active monitoring |
| `MONITORING` | Camera feed is active, CV pipeline running |
| `ANALYZING` | VLM inference in progress |
| `ALERTING` | Sending an alert notification |
| `ERROR` | Recoverable error state |

Defined in `agents/mcp/state_machine.py`. Transitions are thread-safe (RLock) and audited.

---

## MCP Tool System

### Two Domains

| Domain | Interpreter | Executor | Tools |
|--------|-------------|----------|-------|
| **general** | `MCPInterpreter` | `MCPExecutor` | Session/state control, frame analysis, alerts, evidence, history |
| **pcb** | `PCBInterpreter` | `PCBExecutor` | PCB inspection, frame-store workflow, defect analytics, reporting, notification preferences |

### Domain Routing

1. User message arrives via SocketIO → `chat.py`
2. `MCPManager.interpret()` calls the **LLM intent classifier** which returns `{domain, tool, confidence, params}`
3. Routes to the correct domain's interpreter
4. Interpreter produces an `MCPToolCallProposal`
5. `MCPManager.submit()` sends it to the domain's executor
6. Executor checks confirmation policy → auto-execute or wait for user approval

### Tool Call Lifecycle

```
User Message
    │
    ▼
MCPToolCallProposal
  state: PROPOSED → PENDING_APPROVAL → APPROVED → EXECUTING → SUCCEEDED/FAILED
                                    ↘ REJECTED (user declines)
```

Defined in `agents/mcp/lifecycle.py`. Every transition is timestamped and audit-logged.

### PCB Domain Tools

| Tool | Confirmation | Description |
|------|:---:|-------------|
| `get_latest_pcb_frames` | No | List stored PCB frames from the frame store |
| `inspect_pcb_frame` | No | Send a stored frame to VLM for defect analysis |
| `inspect_pcb` | No | Analyze the live camera frame |
| `classify_board` | No | Identify board type from current frame |
| `send_defect_alert` | **Yes** | Email defect alert to recipients |
| `log_defect` | No | Record defect to database |
| `generate_defect_report` | No | Aggregate defect summary |
| `start_defect_monitoring` | No | Deprecated — monitoring loop + LLM handle this |
| `stop_defect_monitoring` | No | Deprecated — see above |
| `query_pcb_inspections` | No | Query past inspection history (PASS/FAIL) |
| `get_monitoring_status` | No | Monitoring status + recent activity summary |
| `toggle_email_notifications` | No | Enable/disable notifications and set thresholds/recipients |
| `get_defect_summary` | No | Defects in a time range (optional filters) |
| `count_defective_pcbs` | No | Count defective boards over a time window |
| `get_latest_defect` | No | Most recent recorded defect |
| `get_defect_type_breakdown` | No | Frequency by defect type |
| `get_defect_trend` | No | Defect-rate trend analysis |
| `get_most_severe_defect` | No | Highest-severity defect in range |
| `get_top_defect_sources` | No | Boards/sources with highest defect volume |
| `generate_summary_report` | No | Daily/weekly/all-time monitoring report |
| `check_threshold_alerts` | No | Check if defect thresholds are exceeded |
| `get_defect_insights` | No | AI-generated recommendations from defect patterns |
| `get_notification_preferences` | No | Read current notification preferences |

### General Domain Tools

| Tool | Confirmation | Description |
|------|:---:|-------------|
| `start_monitoring_session` | No | Start camera + proactive monitoring |
| `end_session` | No | Stop monitoring, end session |
| `go_idle` | No | Pause monitoring |
| `get_agent_status` | No | Current state + metrics |
| `analyze_current_frame` | No | Analyze live frame via VLM |
| `query_history` | No | Detection log history |
| `get_session_summary` | No | Session statistics |
| `send_alert_email` | **Yes** | Generic alert email |
| `set_detection_task` | No | Configure detection mode |
| `shutdown_agent` | **Yes** | Full shutdown |
| `acknowledge_error` | No | Recover from error state back to idle |

---

## Frame Capture Pipeline (Decoupled from LLM)

The LLM never touches the camera directly. Frame capture is handled by the monitoring loop:

```
Camera Feed Publisher
        │
        ▼
MonitoringLoop                              (agents/core/monitoring_loop.py)
        │
        ├── CV Observation (per frame)
        │     ├── Motion score (ROI diff)
        │     ├── Board-in-zone (contour detection + hysteresis)
        │     └── Board signature (tracker or hash)
        │
        ├── State-change detection
        │     Board just stopped in zone? → on_board_ready()
        │
        └── Frame Store Hook
              │
              │  if board_in_zone AND NOT motion_moving:
              │     save JPEG to disk
              │     INSERT into pcb_frame_store
              │
              ▼
        ┌─────────────────┐
        │ pcb_frame_store │  (SQLite table)
        │  id, timestamp, │
        │  image_path,    │
        │  motion_score,  │
        │  board_signature│
        │  consumed       │
        └────────┬────────┘
                 │
                 │  consumed by:
                 │
        LLM tools (on-demand)
           get_latest_pcb_frames
           inspect_pcb_frame
```

### Automatic Frame Storage

Frames are stored when:
- **Board detected** in the camera's region of interest
- **Motion score < threshold** (board is stationary)
- **Cooldown elapsed** (2 seconds between captures)

Storage path: `$CAMERA_AGENT_DATA_DIR/pcb_frame_store/pcb_{timestamp}.jpg`

---

## How the LLM Handles Defect Monitoring

When the user says *"start monitoring for defective PCBs"*, the system:

1. Starts `MonitoringLoop` which observes the camera feed via CV
2. When a board stops in zone, fires `on_board_ready(frame, context)`
3. The `StreamlinedMonitoringService._on_board_ready` calls the VLM
4. The LLM decides follow-up MCP actions (alerts, analytics queries, reporting, preference updates)

```
MonitoringLoop detects board stopped
        │
        ▼
on_board_ready(frame, observation_context)
        │
        ▼
VLM analysis → DetectionEvent
        │
        ├── Persist to pcb_inspections (PASS/FAIL)
  ├── Auto-log defect side effects when detection tools are used
        ├── Emit SocketIO event → Chat UI
  └── LLM decides: call send_defect_alert / analytics tools?
```

The LLM chains tools together naturally. Runtime monitoring is proactive-only:
`StreamlinedMonitoringService.start_monitoring()` always selects
`MonitoringMode.PROACTIVE`.

---

## Database Schema

SQLite with connection pooling. 8 models, 9 repositories.

### Key Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `pcb_frame_store` | Auto-captured PCB frames | timestamp, image_path, motion_score, board_signature, consumed |
| `pcb_inspections` | VLM inspection outcomes | board_signature, result (PASS/FAIL), confidence, defect_type, image_path, decision_trace |
| `pcb_defects` | Logged defect records | board_type, defect_type, severity, confidence, description |
| `detection_logs` | General detection events | timestamp, confidence, response, image_path, decision_details |
| `users` | Email recipients + settings | email, name, active |
| `chat_messages` | Persisted conversations | client_session_id, message_id, role, content, metadata |
| `notification_preferences` | Alert policy configuration | email_enabled, min_severity, recipients_json, quiet_hours |

### Data Flow

```
Frame captured → pcb_frame_store (image + metadata)
                       │
                       ▼
              VLM inspection
                       │
                       ▼
              pcb_inspections (PASS/FAIL record)
                       │
                  if FAIL:
                       ├──▶ pcb_defects (logged defect)
                       └──▶ email alert
```

---

## LLM Integration

### Intent Classification

`agents/classifiers/llm_classifier.py` uses a **single LLM inference call** with structured JSON output:

```json
{
  "domain": "pcb",
  "tool": "count_defective_pcbs",
  "confidence": 0.95,
  "params": {"hours": 24},
  "rationale": "User asked how many defective boards were found today"
}
```

The classifier prompt embeds all available tools with descriptions. A simple circuit breaker (3 consecutive failures → 30s backoff) handles vLLM downtime.

### VLM (Vision Language Model)

`agents/vlm/client.py` — `UnifiedVLMClient` supports:
- **vLLM backend** (OpenAI-compatible `/v1/chat/completions`)
- **Ollama backend** (local `/api/generate`)

Used for frame analysis: encodes images as base64, sends with task-specific prompts, returns structured detection results.

### LLM Router

`router/llm_router.py` — single-provider vLLM router. Handles conversational responses (non-tool chat messages) via the same vLLM deployment. Tracks token usage.

---

## Chat Interface

### Backend event flow (`app/websocket/chat.py`)

1. **`chat_message`** → `_process_user_message()` → MCPManager interpret/submit.
2. Pending proposals emit **`tool_confirmation_required`**.
3. User emits **`approve_proposal`** or **`reject_proposal`**.
4. Server emits **`proposal_result`**, **`chat_message`**, and (when needed) **`agent_state_changed`**.

Session management uses `ChatSessionManager` with persisted message history (`chat_messages` table) keyed by stable `client_session_id`.

### Frontend runtime (`templates/base.html` + `templates/chat.html`)

- `base.html` owns Socket.IO initialization and translates key socket events into browser `CustomEvent`s.
- `chat.html` listens to `agent-chat-connected`, `agent-chat-message`, and `agent-state-changed` to avoid duplicate `socket.on(...)` registration.
- Additional chat-specific socket handlers in `chat.html` include `tool_confirmation_required`, `proposal_result`, `conversation_cleared`, `chat_detection`, `new_log`, and `agent_activity`.

### Camera + status behavior in chat view

- Camera stream starts when agent state indicates active monitoring (`monitoring`, `analyzing`, `alerting`) or `monitoring_active` is true.
- Live image source uses `/api/video_feed` with periodic metadata/stat refresh from `/api/capture_frame`.
- Status fallback path calls `/api/status` (and `/api/v1/tools`) when socket bootstrap does not arrive.
- Offline mode remains interactive: user messages get local mock responses while preserving UI responsiveness.

---

## Typical User Flows

### Flow 1: Start Monitoring

```
User: "Monitor for defective PCBs and alert admin@acme.com"
  │
  ├─ Classifier: domain=general, tool=start_monitoring_session
  ├─ MonitoringLoop starts, observes camera feed
  │
  ...board stops in zone...
  │
  ├─ on_board_ready → VLM analysis
  ├─ Defect detected → LLM calls send_defect_alert
  │
  ▼
Agent: "🚨 Defect detected — solder bridge. Alert sent to admin@acme.com"
```

### Flow 2: Ask About Past Inspections

```
User: "How many defective PCBs did you find today?"
  │
  ├─ Classifier: domain=pcb, tool=query_pcb_inspections
  ├─ Interpreter: proposal(limit=10, result_filter="FAIL")
  ├─ Executor: auto-approved
  │
  ▼
Agent: "Found 47 total inspections. Showing 10 most recent. Pass: 3, Fail: 7."
```

### Flow 3: On-Demand Inspection

```
User: "Check the latest PCB for defects"
  │
  ├─ Classifier: domain=pcb, tool=inspect_pcb_frame
  ├─ Interpreter: proposal(frame_id=None → latest unconsumed)
  ├─ Executor: auto-approved → VLM analysis
  │
  ▼
Agent: "✅ Stored PCB frame inspected — solder bridge detected, severity: high"
  │
User: "Send an alert about that to ops@acme.com"
  │
  ├─ Classifier: domain=pcb, tool=send_defect_alert
  ├─ Interpreter: proposal(recipients=["ops@acme.com"], ...)
  ├─ Executor: requires_confirmation
  │
  ▼
Agent: "🔔 Send a PCB defect alert email to ops@acme.com?"
User: [Approve]
Agent: "✅ Defect alert sent to 1 recipient(s)"
```

### Flow 4: Stop Monitoring

```
User: "Stop monitoring"
  │
  ├─ Classifier: domain=general, tool=end_session
  ├─ end_session handler:
  │    ├─ stops MonitoringLoop
  │    └─ transitions state → IDLE
  │
  ▼
Agent: "✅ Session ended."
```

---

## Deployment

- **Container**: Dockerfile + docker-compose.yml
- **Kubernetes**: Helm chart in `helm/camera-agent/`
- **Runtime**: Gunicorn via `wsgi.py`, SocketIO via eventlet
- **Dependencies**: `requirements/base.txt` (core), `dev.txt`, `prod.txt`, `test.txt`
