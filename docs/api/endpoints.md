# API Reference

## Overview

All REST endpoints are served under the `/api` prefix (v1).

| Module       | Prefix             | Description                       |
|------------- |------------------- |---------------------------------- |
| health       | `/api/health`      | Liveness & readiness probes       |
| monitoring   | `/api/monitoring`  | Start/stop monitoring, status     |
| camera       | `/api/camera`      | Camera feed & snapshots           |
| config       | `/api/config`      | Runtime configuration CRUD        |
| analysis     | `/api/analysis`    | On-demand frame analysis          |
| logs         | `/api/logs`        | Detection & event log queries     |
| users        | `/api/users`       | User management                   |
| system       | `/api/system`      | System metrics (CPU, memory, GPU) |
| mcp          | `/api/mcp`         | MCP tool listing & execution      |
| llm          | `/api/llm`         | LLM router status & testing       |

## Versioning

API blueprints are registered via `app/api/__init__.py:register_api_versions()`.
To add a v2, create `app/api/v2/` and register it with the `/api/v2` prefix.

## WebSocket Events

Socket.IO events are handled in `app/websocket/chat.py`:

| Event           | Direction       | Description                        |
|---------------- |---------------- |----------------------------------- |
| `chat_message`  | client → server | User chat message for agent        |
| `chat_response` | server → client | Agent response                     |
| `frame_update`  | server → client | Live camera frame (base64 JPEG)    |
| `status_update` | server → client | Monitoring status change           |
