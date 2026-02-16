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

| Event | Direction | Description |
| --- | --- | --- |
| `chat_message` | client → server | User chat message for the agent |
| `chat_response` | server → client | Agent response |
| `frame_update` | server → client | Live camera frame (base64 JPEG) |
| `status_update` | server → client | Monitoring status update |
