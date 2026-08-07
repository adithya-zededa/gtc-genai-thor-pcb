# Code Review Findings — 2026-08-06

Snapshot review of the working tree on `UI-Feature-Changes`. Findings are grouped by area and ranked by severity within each group. File:line references point at the reviewed on-disk state, not the uncommitted diff.

## 1. Agent / VLM / MCP core (`agents/`, `router/`, `core/config.py`)

1. **[High] Agentic tool-calling bypasses the proposal/approval pipeline.**
   `agents/core/detection_agent.py:525-528` calls `mcp_executor._invoke(tool_name, arguments)` directly. Every other caller reaches `_invoke` only through `submit_proposal` (`agents/mcp/executor_base.py:200`), which validates input, checks `allowed_in_states`, dedups, and gates on `requires_confirmation`. Since the VLM chooses tool calls from image content plus instructions that tell it to "use THAT EXACT address" (`agents/vlm/prompts.py:228`), text/QR content visible to the camera is a prompt-injection surface that can trigger an unconfirmed tool call (e.g. `send_defect_alert`).

2. **[High] Shared mutable context on the executor singleton, no lock.**
   `agents/core/detection_agent.py:525-527` sets `mcp_executor.context["image_data"]`/`["recipients"]` on the shared `MCPExecutor` singleton immediately before dispatch. Concurrent `analyze_agentic` calls (e.g. two API/session requests in flight) can race and execute a tool call from session A against session B's image or recipients.

3. **[High] SSIM cache can serve a stale result when signature derivation fails.**
   `agents/core/detection_agent.py:230-241` only invalidates the cache when both the new and previous `board_signature` are non-`None`. If `_derive_signature` returns `None` for a genuinely new board, the previous board's cached verdict (possibly "no defect") is reused — in a defect-inspection system this can silently pass a different, defective board.

4. **[Medium] General-domain interpreter has no parameter allowlist.**
   `agents/mcp/domains/general/interpreter.py:84-86` does `arguments.update(result.params)` straight from the LLM classifier's output. The PCB interpreter filters through `TOOL_PARAM_ALLOWLIST` (`agents/mcp/domains/pcb/interpreter.py:117-121`); the general domain has no equivalent guardrail on classifier-influenced fields.

5. **[Medium] No retry on the direct vLLM completion path.**
   `agents/classifiers/llm_classifier.py:404-419` (`_vllm_completion`) issues a single POST with a 15s timeout and no retry; only a global circuit breaker (3 consecutive failures) backstops it, so one flaky response fails the whole `classify()` call.

6. **[Medium] Blocking wait inside the single-threaded monitoring loop.**
   `agents/core/monitoring_loop.py:318-325` calls `self._stop_event.wait(timeout=4.0)` directly in `_loop()`, halting all frame acquisition/CV observation for 4s per "board stopped" event; frames arriving during that window are dropped.

7. **[Medium] Broad `except Exception` swallowing.**
   Pervasive pattern, e.g. `agents/core/detection_agent.py:161-163,197-198,209-210,281-284` and `agents/tools/pcb/_helpers.py:52-53,335-337` — log-and-continue blocks hide root causes (e.g. config-loading bugs masquerading as "missing config").

8. **[Low] Latent `NameError` risk in `agents/vlm/prompts.py:245`.**
   `build_tools_prompt(tools: List[Dict[str, Any]])` uses `List` without importing it (only `Any, Dict, Optional` are imported). Works today only because `from __future__ import annotations` defers evaluation; breaks under `typing.get_type_hints`.

9. **[Low] Duplicate legacy code paths.**
   `DetectionResult`/`analyze_frame` (`agents/vlm/client.py:99-110,602-621`) are marked legacy but still duplicate field-derivation logic (`pcb_count`, `pcb_stable`) with `AnalysisResult`/`analyze`, risking drift between the two.

10. **[Low] Model availability check is advisory only.**
    `agents/vlm/client.py:196-206` (`_ensure_model_available`) only logs a warning if the model isn't found on the vLLM server; construction proceeds regardless, so the failure surfaces later with a less specific error.

## 2. API / services / app layer

No authentication layer exists anywhere in the codebase (`grep -rn "before_request|login_required|require_auth|API_KEY" app/api core` → zero hits). This underlies most items below.

1. **[Critical] Entire API is unauthenticated.** Every route in `app/api/v1/llm.py`, `monitoring.py`, `system.py` (and the rest of `app/api/v1/`) is reachable by anyone with network access.

2. **[Critical] Arbitrary environment-variable injection.** `app/api/v1/system.py:162-176` (`system_environment()`) does `for key, value in data.items(): os.environ[key] = str(value)` with no whitelist. An unauthenticated caller can rewrite `VLLM_URL`, `SPEAKER_DEVICE`, `CAMERA_VIDEO_SOURCE`, etc.

3. **[Critical] SSRF / traffic hijack via LLM config endpoints.** `app/api/v1/llm.py:129-168` (`update_llm_config`) lets an unauthenticated caller repoint the production LLM router at an arbitrary `url`/`api_key`, redirecting all inference traffic (including PCB inspection data). `llm.py:206-252` (`fetch_models_for_provider`) makes the server fetch any caller-supplied `url` — SSRF usable to probe internal network services.

4. **[Bug] `AudioPlayer.is_available()` is broken.** `services/infrastructure/audio.py:211-224` — the docstring accidentally contains real code, and the method body reads `cls._PLAYERS`, a class attribute that's never defined (`_get_player_configs` builds the list as an *instance* method). Any call raises `AttributeError` instead of returning a bool.

5. **[High] Session/CORS weaknesses.** `core/config.py:164-166` regenerates `FLASK_SECRET_KEY` randomly on every process start when unset, breaking signed sessions across multi-worker/restarted deployments. `SOCKETIO_CORS` defaults to `"*"` (`core/config.py:30/94`), so any origin can open a WebSocket and drive the agent given item 1.

6. **[High] Client-controlled `client_session_id` allows cross-session chat access.** `app/websocket/chat.py:301-345` (`get_or_bind_client_session`) and `:225-238` (`_load_messages_from_db`) trust whatever `client_session_id` the browser sends, with no server-issued token — any client can guess/reuse another session's ID and read that user's chat + defect history.

7. **[Medium] Broad file exposure via `serve_image`.** `app/api/v1/logs.py:144-196` allows reading any file under `config.data_dir` (defaults to CWD, so `is_relative_to(data_dir)` is a weak boundary) — can expose `config.yaml` or the SQLite DB file if they live in the working directory.

8. **[Medium] Internal error leakage.** Widespread `except Exception as e: return jsonify({"error": str(e)})` (`llm.py:167-168,250-252`; `system.py:153-154,175-176,189-190`) returns raw exception text to unauthenticated callers.

9. **[Low] Migration errors silently swallowed.** `app/database/connection.py:445-463` (`_apply_migrations`) catches all `sqlite3.OperationalError`, not just "duplicate column" — a locked DB or disk-full error during migration is silently ignored.

10. **[Low] Unauthenticated video-source switch.** `app/api/v1/system.py:257-293` (`video_source()`) lets any caller redirect the live camera feed to an arbitrary local file path, validated only for existence.

## 3. Frontend / config / deployment

1. **[High] Both Helm pods run `privileged: true` unconditionally.** `helm/camera-agent/templates/vllm-deployment.yaml:33-34` and `helm/camera-agent/values.yaml:29-30` grant full host device/kernel access on both the vLLM and the main agent pod, not scoped to specific capabilities.

2. **[High] Plaintext secret fields in `values.yaml`.** `vllmServer.huggingfaceToken`, `vllmServer.ngcApiKey`, `model.huggingface.token`, `model.backendService.bearerToken` are plaintext string fields (empty by default) — one `--set`/CI log away from a leaked credential. `helm/camera-agent/templates/secrets.yaml:37` also builds the dockerconfigjson via `printf` string interpolation, which breaks silently if the key contains `"` or `\`.

3. **[Medium] Dockerfile runs as root, no `.dockerignore`.** `Dockerfile:35` (`COPY . .`) copies the whole repo (including `.git`, any local `.env`) into the image; there's no `USER` directive (runs as root) and no multi-stage build.

4. **[Medium] No resource limits on the main agent pod.** `helm/camera-agent/values.yaml:49-54` sets `requests` but no `limits` for the app container (the vLLM pod does have limits at lines 190-193).

5. **[Low] Tracked binary Helm chart artifact.** `zededa-reference-agent-pcb-thor-vllm-2.15.0.tgz` is committed and edited directly in git rather than produced by a release process — bloats history and makes diffs opaque.

6. **[Low] Inline `onclick` with unescaped interpolation.** `templates/logs.html:699-703` builds `onclick="deleteLog('${log.id}')"` without the `escapeHtml()` treatment used everywhere else in the file — likely safe today (numeric PK) but inconsistent with the rest of the file's escaping discipline.

7. **[Low] Floating image tag, personal namespace.** `helm/camera-agent/values.yaml:6-8` uses `pullPolicy: Always` with a mutable tag (`v57`) from an individual's Docker Hub namespace rather than a pinned digest from an org registry.

8. **[Low] Unescaped `innerHTML` in dashboard activity feed.** `templates/dashboard.html:766-769` (`addActivity()`) interpolates `message` directly into `innerHTML`; currently fed only trusted local strings, but `toggleVideoSource`'s error path threads `data.error` (server-derived) through the same sink.

## Suggested prioritization

1. Add authentication to `app/api/*` and the WebSocket connect handshake (item 2.1) — this is the root cause behind 2.2, 2.3, 2.6, 2.10, and 3's blast radius.
2. Remove/allowlist the env-var injection and LLM-config-repoint endpoints (2.2, 2.3) or gate them behind auth + an explicit admin flag.
3. Fix the tool-dispatch bypass and shared-context race in the agentic path (1.1, 1.2) before relying on `analyze_agentic` under concurrent load.
4. Fix `AudioPlayer.is_available()` (2.4) — trivial, currently broken in all cases.
5. Everything else can follow as normal hardening/cleanup work.
