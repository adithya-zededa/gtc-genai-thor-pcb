# Fixes Implemented — 2026-08-06

Tracks remediation of `docs/reviews/2026-08-06-code-review-findings.md`. Item numbers below match that document's numbering (Section.Item). All 26 findings are addressed except 1.10 and 3.7, which were deliberately left as-is (rationale below).

A second pass then re-reviewed this pass's own diff and fixed four defects it had introduced or left half-done — see "Second pass" below.

Two binding constraints governed every fix in this pass:

1. **No new auth system.** The app remains unauthenticated end-to-end (finding 2.1 is out of scope). Instead, specific dangerous behaviors were hardened directly: env-var allowlisting, LLM-config URL validation, video-source path confinement, and server-issued (rather than client-supplied) chat session tokens.
2. **Deployment/Helm/Docker findings included** in the same pass, not deferred.

## 1. Agent / VLM / MCP core

**1.1 + 1.2 — Tool-dispatch bypass + shared-context race.**
Added `BaseDomainExecutor.submit_agentic_call()` (`agents/mcp/executor_base.py`), which builds an `MCPToolCallProposal` and routes it through the normal `submit_proposal()` pipeline (validation, state-machine gating, dedup, `requires_confirmation`) instead of calling `_invoke()` directly. Context mutation (`image_data`, `recipients`) now happens under a new `self._context_lock`, closing the race between concurrent `analyze_agentic` calls. `detection_agent.py`'s `tool_dispatch` closure now calls `submit_agentic_call(...)`, and a new `_adapt_agentic_result()` guarantees a top-level `"success"` key for every outcome (`executed`/`pending_approval`/`rejected`/`failed`), since `UnifiedVLMClient.analyze_with_tools` defaults `success=True` when the key is absent.
**Behavior change:** `send_alert_email` (the only agentic tool with `requires_confirmation=True`) now surfaces as a pending-approval prompt instead of firing immediately from the VLM tool loop. This is the intended effect of routing through the approval pipeline, not a regression.

**1.3 — SSIM cache stale-result bug.**
`_check_ssim_skip` (`agents/core/detection_agent.py`) now treats a `None` new signature (signature derivation failed) as an unconditional cache invalidation — a board with no derivable signature can no longer reuse a previous board's cached "no defect" verdict.

**1.4 — General-domain interpreter had no parameter allowlist.**
Added `TOOL_PARAM_ALLOWLIST` to `agents/mcp/domains/general/tool_defs.py`, keyed per tool (mirroring the PCB domain's existing pattern). `agents/mcp/domains/general/interpreter.py` now filters `result.params` through it before merging into `arguments`, instead of `arguments.update(result.params)` unfiltered.

**1.5 — No retry on the direct vLLM completion path.**
`_vllm_completion` (`agents/classifiers/llm_classifier.py`) now retries once (2 attempts, 0.5s backoff) on `ConnectionError`/`Timeout` only — not on HTTP 4xx, which re-raises immediately.

**1.6 — Blocking 4s wait halted frame acquisition.**
`MonitoringLoop._loop()` (`agents/core/monitoring_loop.py`) replaced the single `self._stop_event.wait(timeout=4.0)` with a short-interval poll loop that keeps calling `_acquire_frame()`/`_observe()` during the settle window, so `frames_processed`/`last_observation`/motion tracking stay current instead of freezing for 4s. The 4s settle delay before inspection capture is preserved; single-threaded design is unchanged (no new threads, no new races).

**1.7 — Broad `except Exception` swallowing.**
Audited each flagged site. Where a specific exception type was obvious, narrowed it (e.g. `agents/tools/pcb/_helpers.py`'s config-loading fallback now catches `(OSError, ValueError, yaml.YAMLError)` instead of bare `Exception`). Sites that must stay broad now log via `logger.exception(...)` (full traceback) instead of `logger.warning`/`logger.error` with just the message, so root causes are no longer silently masked as expected fallbacks (`detection_agent.py`'s image-save/analysis-failure paths, `_helpers.py`'s defect-summary auto-generation).

**1.8 — Missing `List` import.**
Added `List` to the `typing` import in `agents/vlm/prompts.py`.

**1.9 — Duplicate `DetectionResult`/`AnalysisResult` field derivation.**
`agents/vlm/client.py`'s legacy `analyze_frame` path now reads `pcb_count`/`pcb_stable` directly off the (already-computed) `AnalysisResult` fields instead of re-deriving them from `result.details`.

**1.10 — `_ensure_model_available` advisory-only check — left as-is.**
The factory (`services/infrastructure/vlm.py`) intentionally passes `wait=False` for deferred/soft-check semantics, tolerating vLLM warm-up delays at startup. Forcing fail-fast risks breaking startup reliability. No-op, documented here as a deliberate decision rather than an oversight.

## 2. API / services / app layer (no-auth hardening)

**2.2 — Arbitrary env-var injection.**
`app/api/v1/system.py` added `ALLOWED_ENV_VARS = frozenset({"CAMERA_INDEX", "DIFF_THRESHOLD", "CAPTURE_INTERVAL", "LOG_LEVEL"})`. `system_environment()`'s POST branch now rejects (400) any key not in the allowlist instead of writing every posted key straight into `os.environ`.

**2.3 — SSRF via LLM config endpoints.**
Added `_validate_provider_url()` (`app/api/v1/llm.py`): rejects non-`http(s)` schemes and a blocklist of cloud metadata hosts (`169.254.169.254`, `fd00:ec2::254`, `metadata.google.internal`). Applied to both `update_llm_config()` and `fetch_models_for_provider()` before the URL is used. This is a partial mitigation by design — private/loopback hosts are intentionally *not* blocked, since the app's own vLLM target is legitimately a private address; the fix closes the highest-value SSRF target (cloud credential theft) without breaking the primary use case.

**2.4 — `AudioPlayer.is_available()` broken.**
`services/infrastructure/audio.py` — the method body had literally been left as commentary/dead code inside the docstring (an `AttributeError`-guaranteed bug: `cls._PLAYERS` was never defined). Replaced with the working implementation: instantiate the class, call `_get_player_configs()`, return `any(shutil.which(player) for player, _ in player_configs)`.

**2.5 — Secret-key regeneration + CORS wildcard.**
- `core/config.py` added `_get_or_create_secret_key()`: persists an auto-generated secret to `<data_dir>/.secret_key` (0600 perms) and reuses it across restarts, instead of calling `os.urandom(...)` fresh every process start (which invalidated signed sessions/tokens on every restart). `.secret_key` is now also added to `.gitignore` (see Documentation/gap notes below).
- `DEFAULT_SOCKETIO_CORS` changed from `"*"` to `None`; `app/__init__.py` now passes `cors_allowed_origins=cors_origins` without the `or "*"` fallback, so flask-socketio's own same-origin-only default applies unless an operator explicitly sets `SOCKETIO_CORS`.

**2.6 — Client-controlled `client_session_id` allowed cross-session chat access.**
`app/websocket/chat.py` now uses `itsdangerous.URLSafeTimedSerializer` (keyed on `config.flask.secret_key`) to issue a signed token the first time a `client_session_id` is bound, returned to the client via a new `session_token` field on the `chat_connected` event. On reconnect, `get_or_bind_client_session()` requires a valid, non-expired (30-day max age) signed token matching the claimed `client_session_id` before reusing that session's history; a missing/invalid/expired token results in a fresh session (new ID + token) rather than binding to someone else's history. `templates/base.html` stores the token in `localStorage` alongside the ID and sends both on connect (`auth: {client_session_id, session_token}`), always adopting whatever the server returns.

**2.7 — Broad file exposure via `serve_image`.**
`app/api/v1/logs.py` dropped the `data_dir`-relative and `Path.cwd()`-relative fallback branches, and the raw absolute-path passthrough branch, entirely. Relative paths now resolve only against `detected_dir`/`processed_dir` (by filename), narrowing exposure from "entire working directory" to just the two intended image directories.

**2.8 — Internal error leakage.**
Every remaining `except Exception as e: return jsonify({"error": str(e)})` site in `app/api/v1/llm.py` and `app/api/v1/system.py` now logs the full exception server-side via `logger.exception(...)` and returns a generic, non-leaking message to the caller. Covered: `update_llm_config`, `list_llm_models`, `fetch_models_for_provider`, `get_llm_token_usage`, `reset_llm_token_usage`, `llm_test_chat`, `system_status`, `system_environment` (GET branch), `test_camera`, `ollama_models`. Deliberately left unchanged: `test_vllm()`'s connectivity-diagnostic message — that endpoint's entire purpose is reporting connectivity failure detail, so it isn't a leak.

**2.9 — Migration errors silently swallowed.**
`app/database/connection.py` replaced three duplicated `try/except sqlite3.OperationalError: pass` blocks with a shared `_add_column_if_missing()` helper that only swallows the specific "duplicate column" error text (case-insensitive substring match, verified against SQLite's actual wording); any other `OperationalError` (locked DB, disk full, corrupt schema) is logged at ERROR and re-raised.

**2.10 — Unauthenticated video-source switch accepted arbitrary path.**
Added `Config.video_dir` property (`core/config.py`, new `CAMERA_VIDEO_DIR` env var, defaults to `Path("video")` — matching the Dockerfile's `/app/video`). `video_source()` (`app/api/v1/system.py`) now resolves the requested path and rejects (400) anything outside `video_dir` via `Path.is_relative_to()`, before checking file existence. The `CAMERA_VIDEO_SOURCE` env-var fallback path is unaffected.

## 3. Deployment / frontend

**3.1 — Both Helm pods `privileged: true` unconditionally.**
`helm/camera-agent/templates/vllm-deployment.yaml`'s pod-level `securityContext` is now templated from `.Values.vllmServer.securityContext` (`{{- toYaml ... | nindent 8 }}`), mirroring the pattern already used for the main app pod. Default value is still `privileged: true` (GPU passthrough requirement Helm can't know per-cluster), but it's now configurable per deployment.

**3.2 — Plaintext secret fields + unsafe dockerconfigjson templating.**
No schema change needed — `secrets.existingSecret` plumbing was already fully wired. Updated the `ngcApiKey` comment in `values.yaml` to match the existing `huggingfaceToken` guidance (don't commit real values; use `--set` or `existingSecret`). Fixed `templates/secrets.yaml:37`'s dockerconfigjson from unsafe `printf "...%s..." | b64enc` string interpolation (breaks on `"`/`\` in the token) to proper `dict ... | toJson | b64enc | quote` templating.

**3.3 — Dockerfile ran as root, no `.dockerignore`.**
`Dockerfile` now creates a non-root `appuser` (UID/GID 1000), `chown`s `/app`, and switches via `USER appuser` before the entrypoint. Paired with a new `fsGroup: 1000` in the Helm chart's `podSecurityContext` (`values.yaml`) so the PVC-mounted `/app/data` volume — root-owned by default without `fsGroup` — stays writable by the non-root container user. Added `.dockerignore` (new file) excluding `.git`, `.env*`, `__pycache__`, `.venv`, `tests/`, `docs/`, `*.tgz`, and other non-runtime assets from the build context. Verified via live `docker build` + `docker run` (confirmed UID 1000, write access to `detected_images/`, `processed_frames/`, `video/`).

**3.4 — No resource limits on the main agent pod.**
Added `limits: {memory: "4Gi", cpu: "2"}` alongside the existing `requests` block in `values.yaml`, matching the style already used for the vLLM pod.

**3.5 — Tracked binary Helm chart artifact.**
`git rm --cached zededa-reference-agent-pcb-thor-vllm-2.15.0.tgz` (file preserved on disk), added a root-anchored `/*.tgz` pattern to `.gitignore`. Deliberately scoped to the root-level file only — the 18 versioned archives under `helm/*.tgz` appear to be an intentional, separate release-archive convention and were left untouched (the finding's wording named only the root-level file, unlike finding 2.8's "widespread" framing).

**3.6 + 3.8 — Unescaped interpolation in `logs.html`/`dashboard.html` — done.**
- `logs.html`: the two `onclick="deleteLog('${log.id}')"` / `openLogDetails` handlers are now `data-log-id` attributes bound with `addEventListener`, matching the `.view-image-btn` pattern already used ten lines below. Escaping alone would *not* have been a correct fix here: an inline handler attribute is HTML-decoded before its contents are parsed as JS, so `escapeHtml()`'s `&#39;` decodes back to `'` and still breaks out of the JS string. Removing the inline handler removes the sink. `data-timestamp` (also unescaped) now goes through `escapeHtml()`.
- `dashboard.html`: added an `escapeHtml()` helper (the file had none) and applied it in `addActivity()`. Extended beyond the finding's wording to the three other `innerHTML` sinks in the same file that interpolate equally untrusted data — `defect_type` (VLM-derived, from the DB), threshold-alert `a.message`, and insight `recommendations`. `renderAgentMemory()` and `renderActivityPlaceholder()` were checked and already safe (`textContent` / static markup).
  Worth noting why this mattered more than "likely safe today": `toggleVideoSource`'s error path renders `data.error`, and `video_source()` echoes the caller-supplied path back in that string (`Video file not found: {source}`) — an unauthenticated request could put arbitrary text in a dashboard `innerHTML`. `handleDetectionEvent` similarly feeds `pcb_analysis_summary` (VLM output, i.e. camera-content-influenced) into the same sink.

**3.7 — Floating image tag, personal namespace — left as follow-up, not fixed.**
Requires a registry/release-process decision (pinning to a digest, moving to an org registry) rather than a code change; out of scope for this pass.

## Incidental fixes caught while documenting this pass

- **Restored a route-registration regression in `app/api/v1/system.py`.** An earlier edit in this pass accidentally deleted the `@api_bp.route("/test_inference")` decorator immediately above `def test_inference():`, silently unregistering that endpoint (it became an unreachable, undecorated function). Nothing in the codebase currently calls `/test_inference`, so this had no live impact, but it was a real regression — fixed by restoring the decorator.
- **`.secret_key` (the file introduced by fix 2.5) was not covered by `.gitignore`.** Since `data_dir` defaults to `Path(".")`, this file lands at the repo root in local/dev runs and would have been one `git add -A` away from committing a live secret. Added `.secret_key` to `.gitignore`.

## Second pass — defects found in the first pass's own changes

Re-reviewing the diff above surfaced four problems introduced (or left half-done) by the first pass. All four are fixed.

**A. `serve_image` narrowing broke images in every containerised deployment.**
Fix 2.7 dropped the `if candidate_path.is_absolute()` branch entirely. But `_save_detection_image` persists `str(self.images_dir / filename)`, and `images_dir` comes from `DETECTED_IMAGES_DIR`, which `helm/camera-agent/templates/deployment.yaml:52` sets to the *absolute* `/app/data/detected_images`. Every `log.image_path` in a Helm/Docker deployment is therefore absolute, `paths_to_try` came out empty, and `app/views/__init__.py:70`'s `url_for("api_v1.serve_image", ...)` links 404'd. The absolute-path candidate is restored; the `is_relative_to(detected_dir | processed_dir)` gate — which is the part that actually delivers the finding's narrowing — is unchanged, so the SQLite DB and `config.yaml` under `data_dir` stay unreachable. Verified with a live test client: absolute path, bare filename, and `detected_images/`-prefixed all 200; the DB file, a `../` traversal, and `/etc/passwd` all 404.

**B. The tool-dispatch bypass (1.1/1.2) was fixed at only one of its two call sites.**
`app/api/v1/analysis.py:321-324` had a byte-identical copy of the `mcp_executor.context[...] = ...; return mcp_executor._invoke(...)` pattern behind the `POST /api/analyze_agentic` route — same unvalidated dispatch, same unlocked shared-context mutation, and reachable directly over HTTP rather than only via the monitoring loop. Now routed through `submit_agentic_call()` like the `detection_agent.py` site.

**C. The chat session token (2.6) was bypassable whenever the in-memory map was cold.**
`get_or_bind_client_session()` only demanded a token when `self._client_sessions` already held the claimed id. `ChatSession.__init__` calls `_load_messages_from_db()`, so on a fresh process (restart, or a second worker) an attacker supplying a guessed `client_session_id` and *no* token took the `session is None` branch and had the victim's persisted history rehydrated from SQLite — exactly the `chat.py:225-238` path the finding named. Added `ChatHistoryRepository.has_messages()`; the token is now required when either an in-memory session *or* persisted history exists, and the check fails closed if the DB lookup errors. Verified across six cases including the cold-cache one; first-time binding of a genuinely new id still needs no token.

**D. `docker build` failed from a clean clone.**
The working tree added `*.MOV` to `.gitignore` *and* a hard `mv /app/IMG_1043.MOV /app/video/IMG_1043.MOV` to the Dockerfile. The 258 MB clip is untracked, so it is absent from a fresh clone's build context and the `RUN` aborted the build. The move is now conditional, and `core/config.py` drops `CAMERA_VIDEO_SOURCE` when the path doesn't resolve — without that the image would boot with a dead feed, since `cv2.VideoCapture` on a missing file just fails to open with no fallback to the live camera. Both branches built and run-verified.

Smaller cleanups in the same pass: deleted `tests/unit/agents/test_conveyor_fsm.py` (imported the long-deleted `agents.core.conveyor_inspection_fsm`, which is why every verification run needed `--ignore=`; `pytest tests/unit/` is now clean unqualified); declared `itsdangerous` in `requirements/base.txt` (fix 2.6 imports it directly, previously relying on it as a Flask transitive); removed the unused `Union` import fix 2.10 added to `core/config.py`; stopped `vllm-deployment.yaml` emitting a bare `imagePullSecrets:` (null) key when no pull secrets are configured; moved `secrets.yaml`'s `---` inside its guard so it isn't emitted for an absent NGC key.

## Verification performed

- `pytest tests/unit/ -q` — 21 passed, no `--ignore` needed after the dead test module was deleted. `tests/integration/` and `tests/e2e/` contain no tests (empty besides `__init__.py`), so the previously planned integration run is a no-op — noted here so it isn't mistaken for coverage that exists.
- `python -m compileall` over `app agents core services router` — clean.
- Live smoke tests: `video_source()` path confinement (in-directory / outside-directory / `../` traversal), SQLite migration idempotency + genuine-error propagation, dockerconfigjson templating round-trip (including special characters, verified via a values file rather than `--set` to avoid Helm CLI's own backslash-escaping), `serve_image` (6 path cases), chat session-token binding (6 cases), Flask test-client render of `/` and `/logs`.
- `helm lint` (0 failed) / `helm template` with and without an NGC key.
- `docker build` verified twice — once with the video asset present, once with it excluded to simulate a clean clone — plus `docker run` confirming UID 1000, write access to `detected_images/`/`processed_frames/`, and the live-camera fallback when the clip is absent.

## Not yet done

- Documentation-drift fixes (`docs/reviews/2026-08-06-documentation-drift.md`) remain deferred, per the original request ordering. One correction to that document: it states the real config module is `core/config.py` and that `config/settings.py` (the Pydantic one the deployment guide references) doesn't exist. It does exist — it is simply dead code that nothing imports. It still carries `socketio_cors: str = Field(default="*")`, so it should be deleted rather than documented, or fix 2.5's CORS change will look contradicted by a live-looking file.
- `app/api/v1/analysis.py` still has four `except Exception as e: ... str(e)` responses (finding 2.8's "widespread" framing). `logs.py`'s two were fixed alongside the `serve_image` work; `analysis.py` was left out of scope for this pass.

## Behavioral consequences worth a decision before the next pass

- **Agentic `send_alert_email` now needs a human click.** Routing through `submit_proposal` means the only `requires_confirmation=True` agentic tool returns `pending_approval` to the VLM tool loop and parks a proposal in `_pending_proposals`. Nothing pushes monitoring-loop-originated proposals to a UI (`templates/chat.html` only renders proposals arising from a chat turn), and pending proposals have no expiry, so they accumulate. This only bites when `StreamlinedMonitoringService._agentic_mode` is enabled — it defaults to `False` and the standard path is unaffected — but unattended agentic operation will silently stop emailing. Either give the tool an `auto_approve_policy` for the monitoring-loop caller or surface these proposals in the dashboard.
- **Agentic calls are now subject to the 5s dedup window** (`_DEDUP_WINDOW`) they previously skipped. Two identical tool calls inside one VLM loop now get the second rejected as a duplicate.
- **`_context_lock` is held across tool execution**, i.e. for up to `_invoke_timeout` (60s). That is what makes the shared-context fix correct, but it serialises all agentic tool calls process-wide.
- **The SSRF blocklist is hostname-based** and therefore trivially bypassed by any DNS name that resolves to `169.254.169.254`. The doc above calls it a partial mitigation; the residual gap is specifically DNS, not just private ranges.
