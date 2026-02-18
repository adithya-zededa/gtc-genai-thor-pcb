# API & Route Reference

## Route registration

- API v1 blueprint is mounted at `/api`.
- API v2 blueprint is mounted at `/api/v2` when enabled.
- UI views blueprint is mounted at `/`.

## REST endpoints (v1)

All routes below are relative to `/api`.

### Analysis (`app/api/v1/analysis.py`)

| Method | Path |
| --- | --- |
| POST | `/analyze_prompt` |
| POST | `/analyze_single` |
| POST | `/analyze_agentic` |
| POST | `/analyze_uploaded_image` |

### Camera (`app/api/v1/camera.py`)

| Method | Path |
| --- | --- |
| GET | `/video_feed` |
| GET | `/capture_frame` |

### Config (`app/api/v1/config.py`)

| Method | Path |
| --- | --- |
| GET, POST | `/config` |
| GET | `/config/defaults` |
| POST | `/config/reset` |
| GET, PUT | `/notifications/recipients` |

### Health (`app/api/v1/health.py`)

| Method | Path |
| --- | --- |
| GET | `/health` |
| GET | `/ready` |

### LLM (`app/api/v1/llm.py`)

| Method | Path |
| --- | --- |
| GET | `/llm/providers` |
| GET | `/llm/health` |
| GET | `/llm/status` |
| PUT | `/llm/config` |
| GET | `/llm/models` |
| POST | `/llm/models/fetch` |
| GET | `/llm/usage` |
| DELETE | `/llm/usage` |
| POST | `/llm/chat` |

### Logs (`app/api/v1/logs.py`)

| Method | Path |
| --- | --- |
| GET, DELETE | `/logs` |
| DELETE | `/logs/<int:log_id>` |
| POST | `/logs/export` |
| GET | `/image/<path:image_path>` |
| GET | `/recent_images` |

### MCP (`app/api/v1/mcp.py`)

| Method | Path |
| --- | --- |
| GET | `/tools` |
| GET | `/tools/<tool_name>` |
| GET | `/tools/schemas` |
| POST | `/mcp/interpret` |
| POST | `/mcp/submit` |
| GET | `/mcp/proposals` |
| POST | `/mcp/proposals/<proposal_id>/approve` |
| POST | `/mcp/proposals/<proposal_id>/reject` |
| GET | `/mcp/state` |
| GET | `/mcp/state/transitions` |
| GET | `/mcp/session` |
| GET | `/mcp/audit` |
| GET | `/mcp/audit/metrics` |

### Monitoring (`app/api/v1/monitoring.py`)

| Method | Path |
| --- | --- |
| GET | `/status` |
| POST | `/circuit_breaker/reset` |
| GET, DELETE | `/agent/memory` |
| GET | `/monitoring/proactive/status` |
| POST | `/monitoring/proactive/start` |
| POST | `/monitoring/proactive/stop` |

### System (`app/api/v1/system.py`)

| Method | Path |
| --- | --- |
| GET | `/system/status` |
| GET, POST | `/system/environment` |
| GET, POST | `/system/logging` |
| GET | `/test_camera` |
| GET | `/test_inference` |
| GET | `/test_vllm` |
| GET | `/test_ollama` |
| GET | `/ollama_models` |

### Users (`app/api/v1/users.py`)

| Method | Path |
| --- | --- |
| GET, POST | `/users` |
| DELETE, PUT | `/users/<int:user_id>` |

## UI routes

Routes below are mounted at root (`/`) in `app/views/__init__.py`.

| Method | Path |
| --- | --- |
| GET | `/` |
| GET | `/monitoring` |
| GET | `/chat` |
| GET | `/chat/v2` |
| GET | `/configuration` |
| GET | `/users` |
| GET | `/logs` |
| GET | `/logs/<int:log_id>` |
| GET | `/settings` |

## WebSocket events

Socket.IO events are handled in `app/websocket/chat.py`.

### Chat lifecycle and control

| Event | Direction | Description |
| --- | --- | --- |
| `chat_connect` | client → server | Optional explicit chat init (fallback to auto-init on socket connect) |
| `chat_disconnect` | client → server | Explicitly end chat session |
| `chat_connected` | server → client | Session bootstrap payload (agent state, tools, history, metrics) |
| `chat_error` | server → client | Chat/session initialization or runtime error |

### Conversation and proposal workflow

| Event | Direction | Description |
| --- | --- | --- |
| `chat_message` | client → server | Submit user chat input |
| `chat_message` | server → client | Echo user/assistant/tool messages with updated state |
| `approve_proposal` | client → server | Approve pending proposal for execution |
| `reject_proposal` | client → server | Reject pending proposal |
| `tool_confirmation_required` | server → client | Proposal requires user approval in UI |
| `proposal_result` | server → client | Final result after approval/rejection path |
| `proposal_error` | server → client | Proposal command validation error |
| `agent_activity` | server → client | Real-time activity chips (in-progress/completed/failed) |
| `agent_state_changed` | server → client | Broadcast state transition + available tools/metrics |

### Conversation utility queries

| Event | Direction | Description |
| --- | --- | --- |
| `get_conversation_history` | client → server | Request chat history |
| `conversation_history` | server → client | Returns stored chat messages |
| `clear_conversation` | client → server | Clear chat history for session/client |
| `conversation_cleared` | server → client | Acknowledge successful clear |
| `get_agent_state` | client → server | Query current state/metrics/pending proposals |
| `agent_state` | server → client | Agent state snapshot payload |
| `get_available_tools` | client → server | Query tool list for current state |
| `available_tools` | server → client | Current tool list payload |
| `get_tool_schemas` | client → server | Query JSON schemas for all tools |
| `tool_schemas` | server → client | Tool schema payload |
| `get_pending_proposals` | client → server | Query pending proposals |
| `pending_proposals` | server → client | Pending proposal payload |
| `get_audit_metrics` | client → server | Query audit counters/metrics |
| `audit_metrics` | server → client | Audit metrics payload |

### Monitoring stream events

| Event | Direction | Description |
| --- | --- | --- |
| `subscribe_monitoring` | client → server | Subscribe client to monitoring updates |
| `unsubscribe_monitoring` | client → server | Unsubscribe client from monitoring updates |
| `subscribed` | server → client | Monitoring subscription acknowledgment |
| `unsubscribed` | server → client | Monitoring unsubscription acknowledgment |
| `frame_update` | server → client | Live frame payload (base64 image + metadata) |
| `detection_event` | server → client | Generic detection broadcast |
| `chat_detection` | server → client | Detection rendered as chat message payload |
| `new_log` | server → client | New monitoring/logging entry |
| `monitoring_status` | server → client | Monitoring status update |
| `alert` | server → client | Alert broadcast |

### Client bridge events (browser `CustomEvent`)

`templates/base.html` bridges selected Socket.IO events into browser-level
events consumed by `templates/chat.html`:

- `chat_connected` → `agent-chat-connected`
- `chat_message` → `agent-chat-message`
- `agent_state_changed` → `agent-state-changed`
