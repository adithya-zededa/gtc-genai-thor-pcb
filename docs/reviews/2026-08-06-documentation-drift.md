# Documentation Drift — 2026-08-06

Verified against the current working-tree code (not the uncommitted diff). The recurring theme: an Ollama-removal refactor (`6b27d09` + uncommitted follow-up) landed in code but several docs still describe Ollama as a supported backend.

## `docs/architecture/README.md`

- **Ollama backend claim is stale.** Lines 11, 71, 127, 189, 255 describe `UnifiedVLMClient` as "supporting both vLLM and Ollama backends via the `VLMBackend` enum." `agents/vlm/client.py` has no `VLMBackend` enum and no Ollama code path anymore — only `class UnifiedVLMClient` against a single vLLM backend.
- **"Backward-compatibility modules" no longer exist.** `docs/agent/README.md`'s companion claim (see below) and the architecture doc's directory listing both imply `proactive_monitoring.py` and `resilience.py` still live in `agents/core/` as re-export shims. Neither file exists in `agents/core/` (only `detection_agent.py`, `monitoring_loop.py`, `state.py`, `__init__.py`) — they were deleted, not merely deprecated.
- Everything else checked out: `services/core/monitoring.py:StreamlinedMonitoringService`, `agents/mcp/{manager,lifecycle,audit,globals}.py`, `app/__init__.py`, `templates/users.html` + `User` model, and the Helm chart version (`2.15.0` / appVersion `2.11.0`) all match `Chart.yaml`.

## `docs/agent/README.md`

- **"Backward-Compatibility Modules" table is dead documentation.** Lines ~151-158 list `proactive_monitoring.py` (shim re-exporting `MonitoringLoop`) and `resilience.py` (re-exports `CircuitBreaker`/`CircuitState`) as "still present." Neither file exists on disk. Either they were removed after the doc was written, or the doc describes an aspirational state that never landed — either way the table is now false and should be deleted or moved to the "Deleted Modules" table.
- Everything else (module purposes, `Observation`/`MonitoringContext` dataclasses, `StreamlinedAgent` methods, `state.py` classes `DetectionEvent`/`AgentSnapshot`/`AgentMemory`) matches current code.

## `docs/api/endpoints.md`

- **Missing route:** `/logs/<int:log_id>/diagnostics` (`app/views/__init__.py:95`, renders `log_diagnostics.html`) is not listed in the UI routes table.
- **Ollama routes mischaracterized:** `/test_ollama` and `/ollama_models` (`app/api/v1/system.py:357-381`) are listed as plain routes, but both are now legacy shims that call the vLLM router internally (`get_router().check_health()` / `.list_models()`) — they don't talk to Ollama at all. The doc doesn't flag them as deprecated aliases.
- All other documented path/method pairs verified accurate against the `@api_bp.route(...)` decorators — no other drift.

## `docs/deployment/guide.md`

- **Wrong config module reference.** Lines 56-57 point at `config/settings.py` and describe "Pydantic settings fields." The real file is `core/config.py`, which uses plain `@dataclass` classes — no Pydantic anywhere in it.
- **`docker run` example is missing a required env var.** Lines 13-19's example omits `VLLM_URL`, but `start.sh:15-18` hard-exits if `VLLM_URL` is unset. The documented command will not start the app as written.
- **vLLM sidecar called "optional" when it's required.** Lines 27-28 — `docker-compose.yml` declares `camera-agent` `depends_on: vllm-server`, and the app refuses to boot without a live `VLLM_URL` target (same as above).
- Minor: the `helm install --set` example passes `vllmServer.enabled=true` and `image.tag=v57`, both already the chart defaults — redundant, not wrong.
- Everything else (image repo/tag, `vllmServer.model` default, ports, `requirements/` layout, `config.yaml` default path) matches.

## `docs/development/proactive-monitoring-quickstart.md`

- **Still lists Ollama as a valid backend.** Line 7: "VLM backend running (vLLM **or Ollama**)." `InferenceConfig.backend` is hardcoded to `"vllm"` in `core/config.py`, and no Ollama code path remains — only dead vestiges (`/test_ollama`, `/ollama_models`, a leftover `ollama_client` mock name in `tests/unit/services/test_camera_monitoring.py`).
- **Config defaults table is wrong.** Lines 131/133 list `frame_interval_seconds: 1.5` and `stationary_motion_threshold: 5.0`. Actual defaults (`config.yaml:36,38` and `agents/core/monitoring_loop.py` `DEFAULTS`) are `0.1` and `1.8`.
- **Model name inconsistent across three sources.** The doc's example (lines 9, 39) uses `Qwen/Qwen3-VL-8B-Instruct`; `docker-compose.yml:51` actually serves `Qwen/Qwen3-VL-4B-Instruct`; `helm/camera-agent/values.yaml:116` defaults to `nvidia/Cosmos-Reason2-8B` (with a separate `model.repoId: unsloth/Qwen3-VL-8B-Instruct-GGUF` at line 202). None of the three agree.

## `docs/development/getting-started.md`

- **`config.local.yaml` instructions don't do anything.** Lines 21-23 say to `cp config.yaml config.local.yaml` and edit it, but `core/config.py` only loads a non-default config path via the `CAMERA_AGENT_CONFIG` env var, which the doc never mentions setting. As written, the copied file is silently ignored.
- Setup/test commands (`pip install -r requirements/dev.txt`, `pytest`, `pytest tests/unit/`, coverage flags) all check out against the actual `requirements/` layout and `tests/` structure.

## Summary

| Doc | Drift severity | Root cause |
|---|---|---|
| `docs/architecture/README.md` | Medium | Ollama removal not reflected; dead "compat shim" claim |
| `docs/agent/README.md` | Low-Medium | Dead "compat shim" table |
| `docs/api/endpoints.md` | Low | One missing route, two mischaracterized legacy routes |
| `docs/deployment/guide.md` | Medium | Wrong config module reference, misleading "optional" dependency, missing required env var in example |
| `docs/development/proactive-monitoring-quickstart.md` | Medium-High | Ollama still listed as supported; wrong numeric defaults; inconsistent model names |
| `docs/development/getting-started.md` | Low | Config-override instructions incomplete |

Three of six docs still describe Ollama as a live option after it was removed from the code — that's the single most impactful fix (a search-and-remove pass across `architecture/README.md`, `agent/README.md`, and `proactive-monitoring-quickstart.md`).
