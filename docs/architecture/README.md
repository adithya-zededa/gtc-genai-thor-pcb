# Architecture

This document is the single source of truth for the current system architecture.

## System Summary

The application is a Flask + Socket.IO runtime that provides:

- chat-driven agent control via MCP (Model Context Protocol),
- proactive PCB monitoring from a camera feed,
- VLM-based frame analysis (supporting vLLM and Ollama backends),
- persistence for events, inspections, defects, config, and chat history (SQLite),
- REST APIs for operational and diagnostics workflows,
- server-rendered web UI for dashboard, chat, logs, settings, and user management.

The control surface is language-first: user messages are interpreted into MCP tool proposals, then executed through a proposal lifecycle (including approval for sensitive actions). The system is packaged as a container and deployed via Helm alongside a dedicated vLLM inference server.

## Top-Level Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes (Helm)                     │
│                                                         │
│  ┌──────────────────────┐    ┌────────────────────────┐ │
│  │   camera-agent Pod   │    │      vLLM Server Pod   │ │
│  │   (Flask + SocketIO) │───▶│  (Triton + vLLM)       │ │
│  │   Port 8080          │    │  Port 8000             │ │
│  │   NodePort 30080     │    │  ClusterIP             │ │
│  │                      │    │  GPU: 1x NVIDIA        │ │
│  │  /dev/video0 mount   │    │  Model cache PVC       │ │
│  │  Data PVC (SQLite)   │    └────────────────────────┘ │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

### Internal component flow

```text
User (Web Chat UI / REST API)
      │
      ▼
Flask App (HTTP + Socket.IO)
  - app/__init__.py          (create_app factory)
  - app/websocket/chat.py    (conversational agent loop)
  - app/views/               (dashboard, chat, logs, settings, users)
  - app/api/v1/              (REST endpoints)
      │
      ▼
MCP Manager / Domain Router
  - agents/mcp/manager.py
      │
      ├──▶ General MCP domain
      │      - session lifecycle, status, history, alerts, evidence tools
      │
      └──▶ PCB MCP domain
             - inspection, defect logging, analytics, reporting,
               notification preferences tools
      │
      ▼
Agent State Machine (OFF → IDLE → MONITORING → ANALYZING → ALERTING → ERROR)
  - agents/mcp/state_machine.py

Monitoring Service + MonitoringLoop
  - services/core/monitoring.py     (StreamlinedMonitoringService)
  - agents/core/monitoring_loop.py  (deterministic CV observation)
  - agents/core/detection_agent.py  (StreamlinedAgent – VLM analysis)
      │
      ▼
VLM Client + LLM Router
  - agents/vlm/client.py        (UnifiedVLMClient – vLLM & Ollama)
  - services/infrastructure/vlm.py  (client factory)
  - router/llm_router.py        (AgentLLMRouter – vLLM adapter)
      │
      ▼
Persistence + UI broadcast
  - app/database/*              (SQLite repos & models)
  - Socket.IO events            (real-time UI updates)
```

## Runtime Components

### 1) App and transport layer

- `create_app()` in `app/__init__.py` creates the Flask app, registers API blueprints, view blueprints, and initializes Socket.IO.
- WebSocket chat handlers in `app/websocket/chat.py` drive the conversational agent loop: session management, MCP interpretation, proposal approval/rejection, and real-time activity event emission.
- API v1 routes are registered from `app/api/v1/*` under `/api`, covering health, camera, config, defects, analysis, logs, LLM, MCP protocol, monitoring, system, and users.
- Server-rendered views (`app/views/`) serve the web UI: dashboard (`/`), chat (`/chat`), logs (`/logs`), settings (`/settings`), users (`/users`), and log detail/diagnostics pages.

### 2) Agent state machine

- `AgentStateMachine` (`agents/mcp/state_machine.py`) manages validated state transitions across the agent lifecycle.
- States: `OFF` → `IDLE` → `MONITORING` → `ANALYZING` → `ALERTING` → `ERROR`.
- Transitions are validated against an explicit `VALID_STATE_TRANSITIONS` map; listeners are notified on state change.
- On first WebSocket connection, the agent transitions `OFF → IDLE`.

### 3) MCP orchestration

- `MCPManager` (`agents/mcp/manager.py`) routes each message to either the `general` or `pcb` domain.
- Domain detection uses `LLMIntentClassifier` (`agents/classifiers/llm_classifier.py`) exclusively (no keyword fallback). Falls back to `general` when the LLM is unavailable or classification confidence is below 0.3.
- The classifier includes a per-request TTL cache and circuit breaker (3 consecutive failures → 30s backoff).
- The manager coordinates: domain resolution → interpreter → tool proposal → executor lifecycle.
- If the PCB interpreter returns no proposal, the manager falls back to the general interpreter.
- Global MCP singletons (state machine, audit log, tool registry, executor, interpreter) are provided in `agents/mcp/globals.py`.

### 4) Domain MCPs

Each domain has its own interpreter, executor, and tool registry:

- **General domain** (`agents/mcp/domains/general/`): 13 tools covering session lifecycle (start/end/summary), agent control (status, idle, shutdown, acknowledge error), analysis (current frame), alerting (email), evidence saving, event logging, history queries, and detection task assignment.
- **PCB domain** (`agents/mcp/domains/pcb/`): 24+ tools covering PCB inspection and classification, defect logging, defect analytics (count, trend, breakdown, severity, sources, thresholds, insights), monitoring control, notification preferences, and report generation (defect reports, summary reports).

### 5) MCP proposal lifecycle

- Tool proposals follow the lifecycle: `PROPOSED → PENDING_APPROVAL → APPROVED → EXECUTING → SUCCEEDED | FAILED` (`agents/mcp/lifecycle.py`).
- `BaseDomainExecutor` (`agents/mcp/executor_base.py`) handles shared proposal flow: deduplication (SHA256 hash + 5s window), approval gates for sensitive tools, thread-pool execution, and timeout management.
- `MCPAuditLog` (`agents/mcp/audit.py`) records all lifecycle events with 15 event types, metrics tracking, and sensitive data redaction.

### 6) Monitoring runtime

- `StreamlinedMonitoringService` (`services/core/monitoring.py`) is the monitoring control plane with two modes: `IDLE` and `PROACTIVE`.
- `MonitoringLoop` (`agents/core/monitoring_loop.py`) is the deterministic CV observation loop. It acquires frames, computes motion score, edge density, board-in-zone detection, and board signature via contour hashing.
- On board-ready state change (new board stops in inspection zone), the loop fires an `on_board_ready` callback with a 4-second camera focus delay before capture.
- `StreamlinedAgent` (`agents/core/detection_agent.py`) performs VLM-based PCB inspection with SSIM-based frame deduplication (skips analysis if scene is unchanged), circuit breaker, and retry logic.

### 7) Inference and routing

- `UnifiedVLMClient` (`agents/vlm/client.py`) is the multi-backend VLM client supporting both **vLLM** and **Ollama** backends via the `VLMBackend` enum. Returns typed results: `AnalysisResult`, `AgenticResult`, `DetectionResult`.
- VLM client creation is centralized in `services/infrastructure/vlm.py` (`create_vlm_client_from_config()`) with auto-detection of the available model via `core.model_detect`.
- `AgentLLMRouter` (`router/llm_router.py`) is a vLLM-focused single-provider router using `VLLMAdapter` with token usage tracking (`TokenUsageTracker`).
- VLM prompts are centralized in `agents/vlm/prompts.py`, including the default PCB inspection prompt for Arduino Uno R4 Minima boards.

### 8) Data and persistence

- SQLite database managed through `app/database/connection.py` (`get_db_connection()`, `init_db()`).
- **Models** (`app/database/models.py`): `User`, `DetectionLog`, `ConfigHistory`, `LogSettings`, `PCBDefect`, `PCBFrameStore`, `PCBInspection`.
- **Repositories** (`app/database/repositories.py`): `UserRepository`, `DetectionLogRepository`, `ChatHistoryRepository`, `ConfigHistoryRepository`, `LogSettingsRepository`, `PCBDefectRepository`, `PCBFrameStoreRepository`, `PCBInspectionRepository`.
- Chat history is persisted via `ChatHistoryRepository` within the WebSocket session flow.

### 9) Domain services

- `services/domains/pcb/` provides PCB domain logic: `record_defect()`, `should_alert()`, `generate_defect_report()`, `classify_board_from_analysis()`, `extract_defects_from_analysis()`.
- Alert logic is severity- and board-type-aware, backed by `PCBDefectRepository`.
- `services/domains/pcb/notification_preferences.py` manages per-user notification settings.

## Web Application

The web UI is server-rendered using Jinja2 templates (`templates/`) with static assets (`static/css/`):

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `dashboard.html` | System overview dashboard |
| `/chat` | `chat.html` | Conversational agent interface (Socket.IO) |
| `/logs` | `logs.html` | Detection log listing |
| `/logs/<id>` | `log_detail.html` | Individual log detail view |
| `/logs/<id>/diagnostics` | `log_diagnostics.html` | Log diagnostics view |
| `/settings` | `settings.html` | Camera, VLM, notification configuration |
| `/users` | `users.html` | User management |

Legacy routes (`/monitoring`, `/chat/v2`, `/configuration`) redirect to their current equivalents.

## Deployment Architecture

### Helm chart (`helm/camera-agent/`)

Chart: `zededa-reference-agent-pcb-thor-vllm` (v2.15.0, appVersion 2.11.0).

The Helm chart deploys **two workloads** into a Kubernetes cluster:

1. **camera-agent** — Flask web application container.
   - Image: `adithyazededa/gtc-genai-thor-pcb`
   - Port 8080 exposed via NodePort (30080)
   - Mounts camera device (`/dev/video0`) and optional speaker (`/dev/snd`)
   - Data PVC for SQLite database and detected images
   - Connects to vLLM server via cluster-internal URL

2. **vLLM server** — GPU-accelerated inference server.
   - Image: `nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3`
   - Port 8000 via ClusterIP (cluster-internal)
   - 1x NVIDIA GPU with configurable memory utilization
   - Default model: `nvidia/Cosmos-Reason2-8B` (configurable)
   - HuggingFace model cache PVC (50Gi)
   - Configurable tensor parallelism, prefix caching, max model length

### Docker Compose (`docker-compose.yml`)

For local development, Docker Compose deploys the same two-service topology:

1. **camera-agent** — builds from `Dockerfile`, port 8080, camera device passthrough, config/data/DB volumes.
2. **vllm-server** — `nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3`, port 8000, 1x GPU reservation, model cache volume. Default model: `Qwen/Qwen3-VL-4B-Instruct` (smaller than Helm default for local dev).

### Entry points

- `run.py` — CLI entry point: loads env → config → logging → model auto-detection → LLM router init → database init → `create_app()` → `socketio.run()`.
- `wsgi.py` — WSGI entry point for gunicorn with eventlet worker.
- `start.sh` — Container entrypoint script.

## Primary Execution Flows

### Chat command flow

1. User sends a chat message via Socket.IO.
2. `MCPManager.detect_domain()` classifies intent via the LLM classifier → `general` or `pcb`.
3. Domain interpreter creates an `MCPToolCallProposal`.
4. Executor runs proposal lifecycle: deduplication check → approval gate (if required) → tool execution.
5. Tool result is emitted back to the chat session, persisted to the database, and recorded in the audit log.

### Proactive PCB monitoring flow

1. Monitoring is started via MCP tooling (`start_monitoring_session` / `start_defect_monitoring`) or the proactive API endpoint.
2. `MonitoringLoop` subscribes to camera frames and computes observations (motion, edge density, board presence, board signature).
3. On board-ready state change (new board detected, motion stopped, 4s focus delay elapsed), `on_board_ready` callback fires.
4. `StreamlinedAgent` performs VLM analysis with SSIM dedup; inspection/defect results are persisted.
5. Results are broadcast via Socket.IO; follow-up actions (alerts/reports/preferences) are handled through MCP tools.

## Key Directories

```text
agents/
  classifiers/      # LLM domain/intent classification with cache + circuit breaker
  core/             # MonitoringLoop, StreamlinedAgent, state models, AgentMemory
  mcp/              # MCP lifecycle, manager, registries, state machine, audit log
    domains/
      general/      # General domain interpreter, executor, tool definitions (13 tools)
      pcb/          # PCB domain interpreter, executor, tool definitions (24+ tools)
  tools/            # Tool handler implementations
    general/        # Alert, evidence, event log, history handlers
    pcb/            # Inspection, analytics, defect logging, reporting, notification handlers
  vlm/              # UnifiedVLMClient (vLLM + Ollama), prompts, task types

app/
  api/v1/           # REST endpoints (health, camera, config, defects, analysis,
                    #   logs, LLM, MCP, monitoring, system, users)
  database/         # SQLite models, repositories, connection management
  views/            # Server-rendered view routes
  websocket/        # Socket.IO chat handlers, session management

services/
  core/             # StreamlinedMonitoringService, camera publisher, inference
  domains/
    pcb/            # PCB defect recording, alerting, report generation
  infrastructure/   # VLM client factory, camera/config utilities

router/             # AgentLLMRouter, VLLMAdapter, token usage tracking
config/             # Pydantic settings/schemas, runtime defaults
core/               # Shared config, logging, errors, model detection, resilience
helm/               # Kubernetes Helm chart (camera-agent + vLLM server)
templates/          # Jinja2 HTML templates (dashboard, chat, logs, settings, users)
static/             # CSS and static assets
```

## Current Design Decisions

- Domain-separated MCP execution (`general` vs `pcb`) is retained as the primary organization boundary.
- Monitoring observation (deterministic CV) and LLM/tool decisioning are intentionally separated.
- The LLM router is currently optimized for vLLM deployment rather than multi-provider routing; the VLM client additionally supports Ollama for alternative deployments.
- The architecture supports both conversational operation (Socket.IO chat) and operational API access (`/api/*`).
- The Helm chart co-deploys the app and inference server as separate pods, keeping GPU resources isolated to the vLLM server.
- SQLite is used for persistence to keep the deployment self-contained (no external database dependency).

## Scope of this document

This file intentionally replaces older architecture docs in this folder. Keep this README updated when changing runtime boundaries, module responsibilities, or cross-component flow.
