# Architecture

This document is the single source of truth for the current system architecture.

## System Summary

The application is a Flask + Socket.IO runtime that provides:

- chat-driven agent control via MCP (Model Context Protocol),
- proactive PCB monitoring from a camera feed,
- VLM-based frame analysis against a dedicated vLLM inference server,
- persistence for events, inspections, defects, config, and chat history (SQLite),
- REST APIs for conversational, operational, and diagnostics workflows,
- server-rendered web UI for dashboard, chat, logs, settings, and user management.

The control surface is language-first: user messages are interpreted into MCP tool
proposals, then executed through a proposal lifecycle (including approval for
sensitive actions). The system is packaged as a container and deployed via Helm
alongside two vLLM inference servers.

## Top-Level Architecture

The deployment serves **two models on two pods** sharing one physical GPU:

```text
┌───────────────────────────────────────────────────────────────────────┐
│                          Kubernetes (Helm)                            │
│                                                                       │
│  ┌──────────────────────┐    ┌────────────────────────────────────┐   │
│  │   camera-agent Pod   │───▶│   vLLM "vision" Pod                │   │
│  │   (Flask + SocketIO) │    │   LFM2.5-VL-1.6B-PCB-Inspect       │   │
│  │   Port 8080          │    │   Port 8000 · ClusterIP            │   │
│  │   NodePort 30080     │    │   gpuMemoryUtilization 0.30        │   │
│  │                      │    └────────────────────────────────────┘   │
│  │  /dev/video0 mount   │    ┌────────────────────────────────────┐   │
│  │  Data PVC (SQLite    │───▶│   vLLM "agent" Pod                 │   │
│  │    + images)         │    │   LFM2.5-2.6B (text)               │   │
│  │  replicas: 1 (fixed) │    │   Port 8000 · ClusterIP            │   │
│  └──────────────────────┘    │   gpuMemoryUtilization 0.25        │   │
│                              └────────────────────────────────────┘   │
│                                     ▲                                 │
│                        1× NVIDIA GPU, device-plugin time slicing      │
└───────────────────────────────────────────────────────────────────────┘
```

**The two models never overlap in job.** The vision model only ever looks at
frames and fills in a structured defect verdict (`agents/vlm/schemas.py`). The
agent model classifies intent, writes chat prose, and selects tools; it never
sees pixels.

### Internal component flow

```text
User (Web Chat UI · Socket.IO)          Operator / script (REST)
      │                                          │
      ▼                                          ▼
Flask App (app/__init__.py — create_app factory)
  - app/websocket/chat.py    (Socket.IO transport + event sink)
  - app/api/v1/chat.py       (REST conversational endpoint)
  - app/api/v1/*             (REST protocol + ops endpoints)
  - app/views/               (dashboard, chat, logs, settings, users)
      │
      │  both transports call the same orchestrator
      ▼
Conversation layer  (agents/conversation/)
  - orchestrator.py   (ConversationOrchestrator — the turn)
  - events.py         (ConversationEventSink — progress/approval side channel)
  - session.py        (ChatSession, bounded session cache, signed tokens)
  - responses.py      (agent-model prose + deterministic fallbacks)
      │
      ▼
MCP Manager / Domain Router  (agents/mcp/manager.py)
      │
      ├──▶ General MCP domain — session lifecycle, agent control, analysis,
      │      alerting, evidence, event log, history  (13 tools)
      │
      └──▶ PCB MCP domain — inspection, defect logging, analytics, monitoring
             control, notification preferences, reporting  (22 tools)
      │
      ▼
Agent State Machine (OFF → IDLE → MONITORING → ANALYZING → ALERTING → ERROR)
  - agents/mcp/state_machine.py
      │
      ▼
Monitoring runtime
  - services/core/monitoring.py     (StreamlinedMonitoringService — control plane)
  - agents/core/monitoring_loop.py  (MonitoringLoop — deterministic CV)
  - agents/core/detection_agent.py  (StreamlinedAgent — VLM inspection)
      │
      ▼
Inference
  - agents/vlm/client.py            (UnifiedVLMClient — vision role)
  - router/llm_router.py            (AgentLLMRouter — one instance per role)
      │
      ▼
Persistence + UI broadcast
  - app/database/*                  (SQLite repos, models, retention sweep)
  - Socket.IO events                (real-time UI updates)
```

## Runtime Components

### 1) App and transport layer

- `create_app()` in `app/__init__.py` creates the Flask app, registers API and view
  blueprints, and initializes Socket.IO.
- **Concurrency model: Socket.IO `threading` mode.** Neither eventlet nor gevent is
  installed, so Flask-SocketIO selects the threading backend, and `run.py` serves it
  through Werkzeug with `allow_unsafe_werkzeug=True`. Alongside Flask's request
  threads the process runs the camera capture thread, the `MonitoringLoop` thread,
  the retention worker, and one `ThreadPoolExecutor` per MCP domain. Blocking calls
  (OpenCV, `requests` to vLLM, SQLite) are therefore safe — they occupy a thread, not
  an event-loop hub.
- **`replicaCount: 1` is a hard requirement, not a default.** SQLite on a
  ReadWriteOnce PVC, the process-wide singletons in `agents/mcp/globals.py`, the
  in-memory session cache, the `/dev/video0` host device, and Socket.IO with no
  message queue all assume a single process. Scaling out requires replacing all five.
- API v1 routes are registered from `app/api/v1/*` under `/api`, covering chat,
  health, camera, config, defects, analysis, logs, LLM, MCP protocol, monitoring,
  system, and users.
- Server-rendered views (`app/views/`) serve the web UI.

### 2) Conversation layer (`agents/conversation/`)

The chat turn is transport-independent. `ConversationOrchestrator.handle_turn()`
interprets a message, routes it to a domain MCP, submits the proposal, and returns
every message the turn produced — including the natural-language follow-up that
summarises a tool result.

It emits nothing itself. Progress updates, approval prompts, and state-change
notifications go to a `ConversationEventSink` supplied by the caller:

| Transport | Sink | Behaviour |
|---|---|---|
| Socket.IO (`app/websocket/chat.py`) | `SocketIOEventSink` | Emits `agent_activity`, `tool_confirmation_required`, `chat_message`, `agent_state_changed`, scoped to the originating socket's room |
| REST (`app/api/v1/chat.py`) | default (no-op) | Progress is dropped; the messages come back in the response body |

`approve()` and `reject()` are separate entry points because a confirmation-gated
tool suspends the turn — the approval arrives as a later, independent request.

### 3) Agent state machine

- `AgentStateMachine` (`agents/mcp/state_machine.py`) manages validated transitions.
- States: `OFF` → `IDLE` → `MONITORING` → `ANALYZING` → `ALERTING` → `ERROR`.
- Transitions are validated against an explicit `VALID_STATE_TRANSITIONS` map;
  listeners are notified on state change.
- On first WebSocket connection, the agent transitions `OFF → IDLE`.

### 4) MCP orchestration

- `MCPManager` (`agents/mcp/manager.py`) routes each message to the `general` or
  `pcb` domain.
- Domain detection uses `LLMIntentClassifier` (`agents/classifiers/llm_classifier.py`)
  exclusively — no keyword fallback. It runs on the **agent** model. Falls back to
  `general` when the LLM is unavailable or confidence is below 0.3.
- The classifier has a per-request TTL cache and a circuit breaker (3 consecutive
  failures → 30s backoff).
- If the PCB interpreter returns no proposal, the manager falls back to the general
  interpreter.
- Global MCP singletons (state machine, audit log, tool registry, executor,
  interpreter) live in `agents/mcp/globals.py`.

### 5) MCP proposal lifecycle

- Proposals follow `PROPOSED → PENDING_APPROVAL → APPROVED → EXECUTING →
  SUCCEEDED | FAILED` (`agents/mcp/lifecycle.py`).
- `BaseDomainExecutor` (`agents/mcp/executor_base.py`) owns the shared flow:
  deduplication (SHA256 hash + 5s window), per-tool parameter allowlists, approval
  gates for sensitive tools, thread-pool execution, and timeouts.
- Agentic tool calls from inside the VLM loop go through `submit_agentic_call()`,
  so they get the same validation and approval gating as chat-initiated calls.
- `MCPAuditLog` (`agents/mcp/audit.py`) records all lifecycle events with 15 event
  types, metrics, and sensitive-data redaction.

**What the approval gate protects.** It stops the *model* from taking a sensitive
action unilaterally. It is not an authorization boundary: the app is unauthenticated
(see [Security posture](#security-posture)), so anyone who can reach the chat can
also reach `POST /api/mcp/proposals/<id>/approve`.

### 6) Monitoring runtime

- `StreamlinedMonitoringService` (`services/core/monitoring.py`) is the control
  plane. It has exactly one runtime: the proactive loop. `MonitoringMode` reports
  `PROACTIVE` while that loop runs and `IDLE` when it does not — these are states,
  not selectable modes, and `is_monitoring` is derived from the loop rather than
  tracked separately.
- `MonitoringLoop` (`agents/core/monitoring_loop.py`) is the deterministic CV
  observation loop. It acquires frames and computes motion score, edge density,
  board-in-zone detection, and a board signature via contour hashing. It owns its
  own camera subscription.
- When a new board settles in the inspection zone, the loop runs an **adaptive focus
  settle**: it keeps acquiring and observing frames throughout a bounded window,
  exits once focus plateaus, and inspects the sharpest in-zone frame it saw. Bounds
  are configurable (`settle_max_seconds`, `settle_min_seconds`, focus plateau and
  improvement thresholds).
- `StreamlinedAgent` (`agents/core/detection_agent.py`) performs the VLM inspection,
  with SSIM-based frame deduplication, a circuit breaker, and retry logic. A frame
  with no derivable signature invalidates the SSIM cache unconditionally, so one
  board can never inherit another's verdict.

**Why CV gates the VLM.** Deterministic vision decides *when* a board is worth
looking at; the VLM is invoked only then. This is what keeps GPU cost proportional
to boards rather than to frames.

### 7) Inference and routing

- `AgentLLMRouter` (`router/llm_router.py`) is instantiated **once per role**,
  `ROLE_VISION` and `ROLE_AGENT`, via `get_vision_router()` / `get_agent_router()`.
  Each role gets its own endpoint, model, timeout, temperature — and its own
  concurrency limiter, so a slow vision inference cannot starve chat of request slots.
- When `AGENT_LLM_URL` is unset both roles resolve to the vision endpoint, so a
  single-pod deployment keeps working unchanged.
- `UnifiedVLMClient` (`agents/vlm/client.py`) is the vision-role client. It returns
  typed results (`AnalysisResult`, `AgenticResult`, `DetectionResult`) and fills the
  structured verdict schema in `agents/vlm/schemas.py`.
- `VLLMAdapter` (`router/adapters/vllm.py`) owns the HTTP call, the retry loop, and
  the concurrency slot. `router/resilience.py` provides only what that path uses:
  `ConcurrencyLimiter`, `calculate_backoff`, `RequestMetrics`, `RequestLogger`.
- VLM prompts are centralized in `agents/vlm/prompts.py`.

### 8) Configuration

`core/config.py` is the **only** place the environment is parsed. Everything else —
the router, the VLM factory, the classifier, the detection agent — reads the
`Config` dataclass tree. Precedence is environment > YAML (`config.yaml`) > default.

| Section | Covers |
|---|---|
| `inference` | Vision model: `VLLM_URL`, `VISION_MODEL`, timeout, temperature, API key |
| `agent_inference` | Agent model: `AGENT_LLM_URL`, `AGENT_MODEL`, `CLASSIFIER_MODEL` |
| `camera` / `database` / `flask` | Device, paths, server |
| `chat` | In-memory session bounds (idle TTL, max sessions, max cached messages) |
| `retention` | Frame-store age/row caps and sweep interval |

`Config` also owns the cross-role fallback rules — `agent_inference_url`,
`agent_inference_model`, `classifier_model`, `resolve_detection_image_dir()` — so
they are stated once instead of re-derived per caller.

`Config` is a cached singleton. Picking up an environment change needs
`reset_config()` **and then** `reset_routers()`; runtime edits via
`AgentLLMRouter.configure()` need neither.

`router/rate_limit_config.py` keeps its own `LLM_*` namespace; it duplicates nothing.

### 9) Data, persistence, and retention

- SQLite via `app/database/connection.py`, WAL mode, 10-connection pool.
- **Models** (`app/database/models.py`): `User`, `DetectionLog`, `ConfigHistory`,
  `LogSettings`, `PCBDefect`, `PCBFrameStore`, `PCBInspection`.
- **Repositories** (`app/database/repositories.py`): one per model, plus
  `ChatHistoryRepository`.
- **Retention** (`app/database/maintenance.py`): a daemon thread started from the
  process entry point sweeps both append-only datasets on an interval. The frame
  store is bounded by age and row count (it is a working buffer); detection logs are
  bounded by the operator-editable `log_retention` setting in days. Both delete the
  backing image files, not just the rows — the data volume is fixed-size, and rows
  alone were never what filled it.

### 10) Domain services

- `services/domains/pcb/` provides `record_defect()`, `should_alert()`,
  `generate_defect_report()`, `classify_board_from_analysis()`,
  `extract_defects_from_analysis()`. Reports are JSON, not PDF.
- Alert logic is severity- and board-type-aware, backed by `PCBDefectRepository`.
- `services/domains/pcb/notification_preferences.py` manages per-user settings.

## Web Application

Server-rendered Jinja2 templates (`templates/`) with static assets (`static/css/`):

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `dashboard.html` | System overview dashboard |
| `/chat` | `chat.html` | Conversational agent interface (Socket.IO) |
| `/logs` | `logs.html` | Detection log listing |
| `/logs/<id>` | `log_detail.html` | Individual log detail view |
| `/logs/<id>/diagnostics` | `log_diagnostics.html` | Log diagnostics view |
| `/settings` | `settings.html` | Camera, VLM, notification configuration |
| `/users` | `users.html` | User management |

Legacy routes (`/monitoring`, `/chat/v2`, `/configuration`) redirect to their
current equivalents.

## Health and probes

Three endpoints with three distinct jobs. They are not interchangeable, and the
Helm chart wires each to the matching probe.

| Endpoint | Probe | Checks | Fails when |
|---|---|---|---|
| `/api/health/live` | `livenessProbe`, `startupProbe` | nothing | the process cannot serve a request |
| `/api/ready` | `readinessProbe` | SQLite | the database is unusable (503) |
| `/api/health` | none — diagnostics | database, camera, **both** inference roles | 503 only if the database is down; degraded otherwise |

Liveness is deliberately dependency-free: a probe that touches the database or a
peer pod turns *their* outage into a restart loop here. Readiness deliberately
excludes inference, because the UI, logs, and history all work while a model is
still loading, and gating the Service on a ten-minute model load would make every
rollout an outage.

`inference_roles` is reported per role (`vision`, `agent`) everywhere it appears.
A reachable vision pod alone is not "inference available" — chat would be dead.

## Deployment Architecture

### Helm chart (`helm/camera-agent/`)

Chart: `zededa-reference-agent-pcb-thor-vllm` (v2.19.0, appVersion 2.11.0).

Three workloads:

1. **camera-agent** — Flask web application.
   - Image: `adithyazededa/gtc-genai-thor-pcb:v57`, runs as non-root UID/GID 1000
   - Port 8080 via NodePort (30080)
   - Mounts `/dev/video0` and optionally `/dev/snd`; `supplementalGroups: [44, 29]`
   - Data PVC (10Gi) for the SQLite database and captured images
   - `replicas: 1` — see [Concurrency model](#1-app-and-transport-layer)

2. **vLLM vision server** — `vllmServer.*`
   - Default model `LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect`, `gpuMemoryUtilization: 0.30`
   - Optional `localModel` to load weights from a node directory instead of the Hub
   - 50Gi model cache PVC, ClusterIP, `strategy: Recreate`

3. **vLLM agent server** — `vllmAgent.*`
   - Default model `LiquidAI/LFM2.5-2.6B`, `gpuMemoryUtilization: 0.25`
   - 30Gi model cache PVC (separate: an RWO PVC cannot be writable by both pods)
   - Set `vllmAgent.enabled=false` to collapse to a single-pod deployment

Both vLLM deployments use `strategy: Recreate` — a rolling update on a single-GPU
node deadlocks waiting for a slice the outgoing pod still holds.

### Docker Compose (`docker-compose.yml`)

Local development runs the app plus **one** vLLM server (`Qwen/Qwen3-VL-4B-Instruct`,
`gpu-memory-utilization=0.85`). With `AGENT_LLM_URL` unset both roles share it.

### Entry points

- `run.py` — CLI entry point: env → config → logging → model auto-detection → LLM
  router init → database init → retention worker → `create_app()` → `socketio.run()`.
- `wsgi.py` — WSGI callable (database init + retention worker at import time).
- `start.sh` — container entrypoint; waits for each configured vLLM endpoint (both,
  when they differ) then execs `run.py`. The wait is non-fatal on timeout: the app
  degrades to "inference unavailable" rather than crash-looping while a model loads.

## Primary Execution Flows

### Chat command flow

1. A message arrives over Socket.IO (`chat_message`) or REST (`POST /api/chat/message`).
2. The transport echoes/audits the user message and calls
   `ConversationOrchestrator.handle_turn()`.
3. `MCPManager.detect_domain()` classifies intent on the **agent** model → `general`
   or `pcb`.
4. The domain interpreter creates an `MCPToolCallProposal`.
5. The executor runs the lifecycle: dedup → parameter allowlist → state gate →
   approval gate (if required) → execution.
6. On success the orchestrator generates a natural-language summary of the tool
   output, grounded in the actual result.
7. Every message is persisted and pushed to the sink; state changes are broadcast.

### Proactive PCB monitoring flow

1. Monitoring starts via MCP tooling (`start_monitoring_session` /
   `start_defect_monitoring`) or the proactive API endpoint.
2. `MonitoringLoop` subscribes to camera frames and computes observations.
3. A new board settles in the zone → adaptive focus settle → `on_board_ready` fires
   with the sharpest frame observed.
4. `StreamlinedAgent` runs the VLM inspection (SSIM dedup, circuit breaker); the
   inspection and any defects are persisted.
5. Results broadcast over Socket.IO; follow-up actions go through MCP tools.

## Security posture

**The application is unauthenticated end-to-end**, and exposed on NodePort 30080.
This is a deliberate reference-deployment trade-off, not an oversight. Specific
behaviours are hardened instead of adding an auth system:

- `POST /api/system/environment` writes are restricted to an allowlist.
- LLM-config URLs are validated against SSRF.
- `serve_image` and the video-source switch confine paths to their roots.
- Chat session tokens are server-issued and signed (`itsdangerous`); a
  `client_session_id` alone cannot claim another client's transcript.
- The Flask secret key persists to `<data_dir>/.secret_key`, so signed tokens
  survive a restart.

What this does **not** protect: every endpoint remains reachable by anyone who can
reach the port, including user management, config writes, and proposal approval.
Deploy behind an authenticating proxy for anything beyond a demo.

## Key Directories

```text
agents/
  classifiers/      # LLM intent classification (agent model) + cache + breaker
  conversation/     # Transport-independent chat turn: orchestrator, sessions,
                    #   event sink, response generation, message type
  core/             # MonitoringLoop, StreamlinedAgent, state models, AgentMemory
  mcp/              # MCP lifecycle, manager, registries, state machine, audit log
    domains/
      general/      # General interpreter, executor, tool definitions (13 tools)
      pcb/          # PCB interpreter, executor, tool definitions (22 tools)
  tools/            # Tool handler implementations
    general/        # Alert, evidence, event log, history handlers
    pcb/            # Inspection, analytics, defect logging, reporting handlers
  vlm/              # UnifiedVLMClient, structured verdict schemas, prompts

app/
  api/v1/           # REST endpoints (chat, health, camera, config, defects,
                    #   analysis, logs, LLM, MCP, monitoring, system, users)
  database/         # SQLite models, repositories, connection pool, retention
  views/            # Server-rendered view routes
  websocket/        # Socket.IO handlers + SocketIOEventSink

services/
  core/             # StreamlinedMonitoringService, camera publisher, inference checks
  domains/pcb/      # Defect recording, alerting, report generation
  infrastructure/   # VLM client factory, camera/config utilities

router/             # Per-role AgentLLMRouter, VLLMAdapter, resilience, token usage
core/               # Config (sole env parser), logging, errors, model detection
helm/               # Kubernetes Helm chart (app + 2 vLLM servers)
templates/          # Jinja2 HTML templates
static/             # CSS and static assets
tools/              # Developer utilities (bench_inspection.py)
archive/            # Retired modules, kept out of the build via .dockerignore
```

## Current Design Decisions

- **Two models on two pods, time-slicing one GPU.** The vision model is a 1.6B
  fine-tune that fills a fixed schema; asking it to also write prose and select
  tools produced worse results at both jobs. The cost is real and should be
  understood: time slicing is *context switching, not concurrency*, so an
  in-flight vision inference adds head-of-line latency to chat and vice versa, and
  the 0.30/0.25 memory split leaves each model less KV cache than it would have
  alone. The alternative — one pod, text-only requests to the VLM — is a supported
  configuration (`vllmAgent.enabled=false`).
- **Deterministic CV gates the VLM.** Motion, edge density, zone occupancy, and
  board signature decide when a board is worth a forward pass.
- **One conversation path, two transports.** The turn lives in
  `agents/conversation/`; Socket.IO and REST differ only in their event sink.
- **Domain-separated MCP execution** (`general` vs `pcb`) is the primary
  organization boundary; a third domain is mechanical to add.
- **`core/config.py` is the sole environment parser.** Modules take configuration,
  they do not re-derive it.
- **SQLite, single replica.** Keeps the deployment self-contained with no external
  database. It is the constraint that fixes `replicaCount: 1`.
- **Bounded by construction.** Chat sessions, cached messages, frame-store rows,
  and detection logs all have explicit limits; the appliance is expected to run
  unattended for weeks on a fixed-size volume.

## Scope of this document

This file intentionally replaces older architecture docs in this folder. Keep it
updated when changing runtime boundaries, module responsibilities, or
cross-component flow.
