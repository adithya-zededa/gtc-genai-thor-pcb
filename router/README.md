# Agent LLM Router

A single-provider router that sends all LLM requests to the project's vLLM
deployment. There is no multi-provider registry, failover, or routing
strategy — `AgentLLMRouter` configures a vLLM connection from
`core.config` and exposes a small `chat()` / `chat_stream()` API on top of it.

There is one router instance **per role**: `vision` (the VLM that looks at
frames) and `agent` (the text model that classifies intent, answers chat, and
selects tools). Each role has its own endpoint, model, and concurrency
limiter. See `get_vision_router()` / `get_agent_router()`.

## What it is (and isn't)

- **Is**: a thin, resilient wrapper around a single OpenAI-compatible vLLM
  endpoint (retries, backoff, concurrency limiting, token-usage tracking,
  streaming).
- **Isn't**: a multi-provider router. There is no `register_provider()`,
  no `RoutingStrategy`, no round-robin/priority/failover selection, and no
  support for Anthropic/OpenAI/Google/Ollama/TGI/LM Studio backends. Only
  vLLM is implemented (`router/adapters/vllm.py`).

## Module layout

| File | Contents |
|------|----------|
| `router/llm_router.py` | `AgentLLMRouter` (one cached instance per role), `get_router(role)`/`get_vision_router()`/`get_agent_router()`/`chat()`, token usage tracking |
| `router/config.py` | `LLMProviderConfig`, `ProviderStatus`, `ChatMessage`, `ChatResponse` dataclasses |
| `router/base.py` | `LLMAdapter` abstract base class shared by all adapters |
| `router/adapters/vllm.py` | `VLLMAdapter` — the only concrete adapter today |
| `router/rate_limit_config.py` | `RateLimitConfig` (retry/backoff/concurrency env vars) |
| `router/resilience.py` | Concurrency limiter, backoff calculator, request metrics/logging |

## Singleton pattern

`AgentLLMRouter` uses a classic double-checked-locking singleton:

```python
class AgentLLMRouter:
    _instance: Optional["AgentLLMRouter"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
```

`__init__` only runs its body once (guarded by `_initialized`), calling
`self._adapter = VLLMAdapter()` and `self._auto_configure()`. Every call to
`AgentLLMRouter()` — or the convenience `get_router()` — returns the same
instance, so configuration changes made via `configure()` are visible
everywhere in the process.

```python
from router import get_router

router = get_router()
response = router.chat(messages=[{"role": "user", "content": "Hello!"}])
```

## Auto-configuration and env vars

On first construction, `AgentLLMRouter._auto_configure()` reads:

| Env var | Used for | Default |
|---------|----------|---------|
| `VLLM_URL` | vLLM server base URL | `http://localhost:8000` |
| `VISION_MODEL` | model name sent to vLLM | `""` (unset → `None`, adapter raises if no model configured at chat time) |
| `VLLM_TIMEOUT` | HTTP request timeout (seconds) | `300` |
| `VLLM_TEMPERATURE` | sampling temperature | `0.1` |
| `VLLM_API_KEY` | optional `Authorization: Bearer` token | unset |

These populate an `LLMProviderConfig(name="vllm", supports_tools=True,
supports_vision=True, ...)` and immediately trigger an availability check
(`_check_availability()` → `VLLMAdapter.check_availability()`), whose result
is stored on `ProviderStatus`.

`core/config.py`'s `RouterConfig` dataclass (part of the app-wide `Config`)
also declares:

| Env var | Field | Default |
|---------|-------|---------|
| `LLM_ROUTER_FOR_CLASSIFICATION` | `router.use_for_classification` | `true` |
| `LLM_ROUTER_FOR_CHAT` | `router.use_for_chat` | `true` |

`router.enabled` is hardcoded `True` (vLLM is the sole provider, so there's
no "disabled" state). Note this is a separate config object from the
router's own env-var reads above — `core.config.InferenceConfig` reads the
same `VLLM_URL`/`VISION_MODEL`/`VLLM_TIMEOUT`/`VLLM_TEMPERATURE` vars
independently for the rest of the app.

Retry/concurrency behavior (used by `VLLMAdapter`, not vLLM-specific) comes
from `router/rate_limit_config.py`'s `RateLimitConfig`:

| Env var | Field | Default |
|---------|-------|---------|
| `LLM_MAX_RETRIES` | `max_retries` | `5` |
| `LLM_BACKOFF_BASE` | `backoff_base` | `2.0` |
| `LLM_BACKOFF_MAX` | `backoff_max` | `30.0` |
| `LLM_BACKOFF_JITTER` | `backoff_jitter` | `0.5` |
| `LLM_MAX_CONCURRENCY` | `max_concurrency` | `2` |
| `LLM_REQUEST_TIMEOUT` | `request_timeout` (seconds to wait for a concurrency slot) | `120.0` |

## `AgentLLMRouter` API

```python
def configure(
    self,
    url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
) -> None: ...

def get_config(self) -> Optional[LLMProviderConfig]: ...

def check_health(self) -> Dict[str, Any]: ...
    # {"vllm": bool, "available": bool, "url": str|None,
    #  "model": str|None, "latency_ms": float|None}

def is_available(self) -> bool: ...

def list_providers(self) -> List[Dict[str, Any]]: ...
    # returns a one-element list (for API compatibility with the old
    # multi-provider shape) containing the vLLM config + status

def list_models(self) -> List[str]: ...

def get_active_provider(self) -> Optional[Dict[str, Any]]: ...

def chat(
    self,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ChatResponse: ...

def chat_stream(
    self,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
):  # generator of SSE-style dict events

def to_dict(self) -> Dict[str, Any]: ...
    # {"provider": "vllm", "config": {...}, "status": {...},
    #  "active_provider": {...}}
```

`chat()` raises `RuntimeError("vLLM provider not configured")` if
`_config` is `None`, and the adapter itself raises `ValueError` if no model
name has been set (`VISION_MODEL` unset and `configure(model=...)` never
called).

`chat_stream()` yields dict events forwarded from the adapter:
`{"type": "token", "content": ...}`, `{"type": "tool_call", "id", "name",
"arguments"}`, `{"type": "done", "response": ChatResponse}`, `{"type":
"error", "error": ...}`. On `"done"`/`"complete"` events the router records
token usage before re-yielding the event.

Module-level convenience functions (`router/llm_router.py`):

```python
def get_router() -> AgentLLMRouter:
    """Get the global LLM router instance."""

def chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> ChatResponse:
    """Send a chat request using the global router."""
```

Token usage is tracked separately from the router instance via a
module-level `TokenUsageTracker`, exposed as `get_token_usage()` /
`reset_token_usage()`.

## Data classes (`router/config.py`)

```python
@dataclass
class LLMProviderConfig:
    name: str = "vllm"
    url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    enabled: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

`url` is normalized in `__post_init__` (adds `http://` if no scheme is
present, strips trailing `/`). `to_dict()` never includes the raw
`api_key` — it exposes `has_api_key: bool` instead.

```python
@dataclass
class ChatResponse:
    content: str
    provider: str
    model: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
```

`ProviderStatus` tracks `available`, `last_check`, `latency_ms`,
`total_requests`, `error_count`, `last_error`, `models_available`.

## `VLLMAdapter` (`router/adapters/vllm.py`)

Implements `LLMAdapter` against vLLM's OpenAI-compatible HTTP API:

- `check_availability()` — `GET {url}/v1/models` (falls back to `GET
  {url}/health` if that doesn't return 200); returns `(available,
  latency_ms, error)`.
- `list_models()` — `GET {url}/v1/models`, returns the `id` field of each
  entry in `data`.
- `chat()` — `POST {url}/v1/chat/completions` with
  `{"model", "messages", "max_tokens", "temperature", "tools"?}`. Adds
  `Authorization: Bearer <api_key>` if `config.api_key` is set. Supports an
  `extra_body` kwarg (merged into the payload — e.g. `chat_template_kwargs`
  for Qwen3 thinking mode). Wrapped in a concurrency-limited retry loop
  (`get_concurrency_limiter()`, `calculate_backoff()`) that retries
  retryable errors (429/5xx/timeouts, see `rate_limit_config.py`) up to
  `max_retries` times with exponential backoff + jitter.
- `chat_stream()` — same endpoint with `"stream": True`, parses the SSE
  `data: {...}` lines, accumulates `delta.content` into token events and
  `delta.tool_calls` into a final `tool_call` event, then yields a `"done"`
  event with the assembled `ChatResponse`.
- `supports_streaming()` returns `True`.

## Adapter base class (`router/base.py`)

`LLMAdapter` is an ABC requiring `check_availability()`, `list_models()`,
and `chat()`. It provides:

- `_get_session()` — a shared `requests.Session` with connection pooling
  (`pool_connections=20`, `pool_maxsize=50`) and a `urllib3` `Retry`
  strategy built from `get_rate_limit_config()`.
- `chat_stream()` — default (non-streaming) implementation that calls
  `chat()` and yields one `{"type": "complete", ...}` event; `VLLMAdapter`
  overrides this with true SSE streaming.
- `supports_streaming()` — defaults to `False`.
- `_convert_tools_to_openai_format()` — converts tool schemas to OpenAI
  `{"type": "function", "function": {...}}` format. `VLLMAdapter.chat()`
  and `chat_stream()` call this when `tools` are passed. Note: this is
  unused by the VLM client's own tool-calling path (`agents/vlm/client.py`
  parses tool calls out of the model's text response rather than using
  structured `tools=`/`tool_calls`), so today it only matters for callers
  that pass `tools=` directly through the router.

## HTTP API (`app/api/v1/llm.py`)

All routes are registered on `api_bp` (see `app/api/v1/__init__.py` for the
blueprint's URL prefix).

### `GET /llm/providers`

Lists the (single) configured provider.

```json
{
  "success": true,
  "enabled": true,
  "providers": [
    {
      "name": "vllm",
      "provider_type": "vllm",
      "url": "http://localhost:8000",
      "model": "Qwen/Qwen3-VL-8B-Instruct",
      "max_tokens": 4096,
      "temperature": 0.1,
      "enabled": true,
      "supports_tools": true,
      "supports_vision": true,
      "has_api_key": false,
      "status": {
        "name": "vllm",
        "available": true,
        "last_check": 1735900000.0,
        "latency_ms": 12.3,
        "total_requests": 4,
        "error_count": 0,
        "last_error": null,
        "models_available": []
      }
    }
  ],
  "active_provider": { "...": "same shape as above" },
  "count": 1
}
```

If the router failed to import, returns `{"success": true, "enabled":
false, "providers": [], "count": 0}` (HTTP 200).

### `GET /llm/health`

```json
{
  "success": true,
  "health": {
    "vllm": true,
    "available": true,
    "url": "http://localhost:8000",
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "latency_ms": 12.3
  },
  "all_healthy": true
}
```

### `GET /llm/status`

Full router state (`AgentLLMRouter.to_dict()`) plus token usage.

```json
{
  "success": true,
  "enabled": true,
  "router": {
    "provider": "vllm",
    "config": { "...": "LLMProviderConfig.to_dict()" },
    "status": { "...": "ProviderStatus.to_dict()" },
    "active_provider": { "...": "config + status" }
  },
  "token_usage": {
    "by_provider": { "vllm/Qwen/Qwen3-VL-8B-Instruct": { "prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165, "request_count": 3 } },
    "totals": { "prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165, "request_count": 3 }
  }
}
```

### `PUT /llm/config`

Runtime update of the vLLM connection (calls `router.configure(...)`).

Request:

```json
{
  "url": "http://llm-service:8000",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "timeout": 300,
  "temperature": 0.1
}
```

Response:

```json
{
  "success": true,
  "message": "vllm configuration updated",
  "config": { "...": "LLMProviderConfig.to_dict()" }
}
```

Returns `400` with `{"success": false, "error": "..."}` on failure.

### `GET /llm/models`

Lists models from the active provider (`router.list_models()`).

```json
{ "success": true, "provider": "vllm", "models": ["Qwen/Qwen3-VL-8B-Instruct"] }
```

### `POST /llm/models/fetch`

Probes an arbitrary URL/adapter without touching the live router config —
useful for a "test connection" UI flow. Uses `router.adapters.get_adapter()`
and a throwaway `LLMProviderConfig(name="_temp_fetch", ...)`.

Request:

```json
{ "url": "http://localhost:8000", "api_key": null, "provider_type": "vllm" }
```

Response:

```json
{ "success": true, "models": ["Qwen/Qwen3-VL-8B-Instruct"] }
```

or, if unreachable: `{"success": true, "models": [], "error": "..."}`.

### `GET /llm/usage` / `DELETE /llm/usage`

`GET` returns `{"success": true, "usage": {"by_provider": {...}, "totals":
{...}}}`. `DELETE` resets counters and returns `{"success": true, "message":
"Token usage reset"}`.

### `POST /llm/chat`

A test/debug endpoint — sends a single user message through the router.

Request:

```json
{ "message": "Hello, what can you do?" }
```

Response:

```json
{
  "success": true,
  "response": {
    "content": "...",
    "provider": "vllm",
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "usage": { "prompt_tokens": 12, "completion_tokens": 30 },
    "finish_reason": "stop"
  }
}
```

Returns `400` if `message` is missing/empty, `500` on router errors.

## Extending

The adapter layer is structurally pluggable — `LLMAdapter` (`router/base.py`)
defines the contract (`check_availability`, `list_models`, `chat`, optional
`chat_stream`/`supports_streaming`), and `router/adapters/__init__.py` keeps
an `_ADAPTER_REGISTRY: dict[str, type[LLMAdapter]]` mapping provider-type
strings to adapter classes, resolved via `get_adapter(provider_type=None)`
(used by `POST /llm/models/fetch`). Today the registry only has:

```python
_ADAPTER_REGISTRY = {
    "vllm": VLLMAdapter,
    "openai": VLLMAdapter,  # OpenAI-compatible API, same wire format
}
```

Adding a new backend means writing a class that subclasses `LLMAdapter` and
adding it to that registry. `AgentLLMRouter`, however, still hardcodes
`self._adapter = VLLMAdapter()` in `__init__` — the registry is only
consulted by the ad-hoc `/llm/models/fetch` probe endpoint today, so wiring
a second *live* provider into `AgentLLMRouter` itself would require
reintroducing some form of provider selection that does not exist now.
