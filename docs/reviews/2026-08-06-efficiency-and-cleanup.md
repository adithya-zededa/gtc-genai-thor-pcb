# Efficiency Review & Repository Cleanup — 2026-08-06

Third pass over the working tree on `UI-Feature-Changes`, focused on **runtime
efficiency** and **repository hygiene**. Complements the two earlier passes
(`2026-08-06-code-review-findings.md` → security/correctness,
`2026-08-06-documentation-drift.md` → docs). Findings here are new; where one
touches something an earlier pass already noted, that is called out.

Timings were measured on this host (Jetson Thor, the deployment target) with
`.venv/bin/python`, 200-iteration loops on a 640×480 BGR frame.

---

## 1. The per-frame pipeline does 3× the work it needs to

`CameraFeedPublisher._capture_loop` ([services/core/camera.py:241-289](../../services/core/camera.py#L241-L289))
runs unconditionally on every captured frame, at the camera's full rate:

| Step | Measured | @30 fps |
|---|---|---|
| `frame.copy()` | 0.043 ms | 1.3 ms/s |
| `_build_overlay_frame` → `OpenCvPcbPresenceDetector.process_frame` | 0.578 ms | 17.3 ms/s |
| `cv2.imencode(".jpg", …)` | 1.280 ms | 38.4 ms/s |
| `base64.b64encode(...)` | 0.025 ms | 0.8 ms/s |
| **total** | **1.93 ms** | **58 ms/s ≈ 5.8% of one core, continuously** |

Three separate problems stack up here.

### 1.1 [High] JPEG + base64 encoding happens whether or not anyone is watching

`image_data` (the JPEG) is consumed **only** by the MJPEG stream
`/api/video_feed` ([app/api/v1/camera.py:18](../../app/api/v1/camera.py#L18)).
`image_b64` is consumed **only** by `/api/capture_frame`, which
[templates/chat.html:2225](../../templates/chat.html#L2225) polls **once per
second**.

So in the normal headless case — proactive monitoring running, no browser
attached — 39 ms/s of the capture thread's 58 ms/s is spent producing bytes
nobody reads. And even with the chat page open, 29 of every 30 base64 strings
are discarded: at 46.9 KB per frame that is **~1.4 MB/s of allocation churn**
thrown away (the CPU cost of b64 itself is negligible; the garbage is the
cost).

Fix: encode lazily. Make `image_data`/`image_b64` cached properties on
`CameraFrame` computed on first access, or gate encoding on
`self._subscribers` being non-empty. `raw_frame` is all the monitoring loop
ever touches.

### 1.2 [High] Two near-identical CV pipelines run on the same frames

`OpenCvPcbPresenceDetector.process_frame`
([services/core/pcb_presence_cv.py:64-125](../../services/core/pcb_presence_cv.py#L64-L125))
and `MonitoringLoop._observe`
([agents/core/monitoring_loop.py:535-616](../../agents/core/monitoring_loop.py#L535-L616))
are the same algorithm written twice: grayscale → crop to the same
`zone_crop_top/bottom_ratio` → resize to 160×120 → `GaussianBlur(5,5)` →
`absdiff` motion score → `Canny(50,150)` edge density → contour segmentation
with the same area/extent/aspect gates → the same presence debounce. Their
default thresholds are identical, down to `stationary_motion_threshold = 1.8`.

They run on different threads at different rates (capture thread @30 Hz,
monitoring loop @10 Hz) with independent temporal state, so the results can
also silently disagree — the UI overlay says "PCB present" while the
monitoring loop says otherwise.

Cost: 0.578 + 0.697 = **1.28 ms per frame pair**, ~17 ms/s of it pure
duplication. Fix: compute presence once in the capture thread and publish the
result on `CameraFrame.metadata` (the field already exists and is already
populated); have `MonitoringLoop._observe` consume it instead of recomputing.
That also collapses the two debounce state machines into one source of truth.

### 1.3 [Low] Frames are dequeued and then thrown away

`MonitoringLoop._acquire_frame` ([agents/core/monitoring_loop.py:370-383](../../agents/core/monitoring_loop.py#L370-L383))
pulls a frame off the subscriber queue and *then* checks the pacing interval,
returning `None` (discarding it) if it arrived too soon. At 30 fps in and a
0.1 s interval, two of every three dequeued frames are discarded. The waste is
small (a queue pop), but the shape is wrong — the check belongs before the
`get_frame` call.

### 1.4 [Low] `_compute_quality` uses double-precision unnecessarily

`_compute_quality` ([monitoring_loop.py:750-758](../../agents/core/monitoring_loop.py#L750-L758))
uses `cv2.Laplacian(roi_blur, cv2.CV_64F)` (0.054 ms) where the neighbouring
`_focus_score` correctly uses `CV_32F` (0.035 ms) for the same measurement.
Small, but it is the same metric computed two different ways in one file.

---

## 2. Chat history grows without bound, and every reconnect replays all of it

Three compounding issues in [app/websocket/chat.py](../../app/websocket/chat.py):

1. **Every progress step is persisted as a chat message.** `_emit_agent_activity`
   ([chat.py:149-180](../../app/websocket/chat.py#L149-L180)) calls
   `chat_session.add_message(...)` for each UI activity update, and
   `add_message` writes straight to SQLite. One user turn produces 2–3 of these
   ("Understanding Request", "Executing Action", "Action Completed") on top of
   the user and assistant messages.

2. **Reconnect loads and ships the entire history.** `_initialize_session`
   calls `chat_session.get_history(limit=0)` ([chat.py:446](../../app/websocket/chat.py#L446)),
   and `get_history` reads `self.messages[-limit:] if limit else self.messages` —
   `limit=0` is falsy, so it returns **everything**. That whole list goes into
   the `chat_connected` payload on every page load. `ChatSession.__init__` →
   `_load_messages_from_db` likewise calls `ChatHistoryRepository.get_messages`
   with no limit.

3. **Sessions are never released.** `ChatSessionManager.remove_session` exists
   but no handler calls it — `handle_disconnect`
   ([app/websocket/\_\_init\_\_.py:53-57](../../app/websocket/__init__.py#L53-L57))
   only logs. Both `_sessions` and `_client_sessions` grow for the process
   lifetime, each holding a full `ChatMessage` list.

Net effect on a long-running edge deployment: chat page load time and payload
size grow linearly with total lifetime conversation, and the process holds
every message ever sent. Fix: don't persist `ui: "activity"` messages (they are
transient UI state, not conversation); pass a real limit at both call sites;
call `remove_session` on disconnect.

---

## 3. Analytics tools fan out into many small queries

[services/domains/pcb/defect_store.py](../../services/domains/pcb/defect_store.py)
composes its higher-level functions out of separate single-purpose queries, so
one tool call becomes many round trips over largely the same rows:

- `tool_count_defective_pcbs` ([analytics.py:204-241](../../agents/tools/pcb/analytics.py#L204-L241))
  issues **four** `COUNT(*)` queries (total, high, medium, low) where one
  `GROUP BY severity` answers all four.
- `get_defect_type_breakdown` runs the `GROUP BY` and then a **second**
  `COUNT(*)` for the denominator, which is just the sum of the groups it
  already has.
- `get_defect_insights` ([defect_store.py:503-590](../../services/domains/pcb/defect_store.py#L503-L590))
  chains `get_defect_trend` + `get_defect_type_breakdown` + `get_top_defect_sources`
  + `check_threshold_alerts` + `count_defects` → **7+ queries**.
- `generate_summary_report` → ~10.
- `tool_get_monitoring_status` → 3 counts + latest + trend + threshold.

On SQLite with a warm pool each query is cheap, so this is a scaling concern
rather than a present-day hotspot — but it is a lot of avoidable work per chat
turn.

**Related, and more clearly wrong:** `get_defect_trend`
([defect_store.py:292-309](../../services/domains/pcb/defect_store.py#L292-L309))
pulls every timestamp in the window into Python and then, *for each row*,
re-parses `bucket["period_start"]` and `bucket["period_end"]` with
`datetime.fromisoformat` inside the inner loop — the same 14 strings parsed
over and over. Measured: 4.0 ms vs 1.7 ms for 5000 rows just from hoisting the
bounds out of the loop (a `GROUP BY` on a bucket expression would remove the
Python loop altogether).

---

## 4. Correctness issues not covered by the earlier passes

### 4.1 [High] Pooled DB connections are returned without rollback

`ConnectionPool.get_connection` ([app/database/connection.py:79-118](../../app/database/connection.py#L79-L118))
returns the connection to the pool in a `finally` block with no
`rollback()`. Every write in `repositories.py` is `conn.execute(...)` followed
by `conn.commit()` — if the `execute` raises (constraint violation, disk
error), the exception propagates, the context manager exits, and the
connection goes back into the pool **with an open write transaction**. The
next borrower of that connection inherits it: it holds SQLite's write lock and
sees a stale snapshot until something else happens to commit.

Fix: `except: conn.rollback(); raise` — or better, make `get_connection` a
transactional context manager that commits on success and rolls back on
exception, and drop the per-call `conn.commit()`s.

### 4.2 [Medium] `pcb_frame_store` grows without bound — on disk and in the DB

`MonitoringLoop._store_frame` ([monitoring_loop.py:766-809](../../agents/core/monitoring_loop.py#L766-L809))
writes a JPEG to `<data_dir>/pcb_frame_store/` and a row to the DB every ≥2 s
while a board is present and stationary. `PCBFrameStoreRepository.cleanup_old`
and `deduplicate_similar` exist to bound that — and **neither has a single
caller** (`grep` across the tree, excluding their own definitions, returns
nothing). On an edge appliance running continuously this fills the data volume.

Either wire `cleanup_old` into the monitoring loop / a periodic task, or drop
both methods (~140 lines) and bound the store at write time.

Note also that `deduplicate_similar` re-reads every candidate image **from
disk** and recomputes similarity — duplicating work `_store_frame` already does
in memory at capture time, with the same 0.92 threshold and 15 s window.

### 4.3 [Medium] The chat UI always reports monitoring as inactive

`StreamlinedMonitoringService.is_monitoring` is never set `True` anywhere —
the proactive runtime tracks state via `proactive_agent` / `_active_mode`
instead. Every consumer uses `get_active_monitoring_mode() != "idle"`
correctly… except two spots in `chat.py`
([:484-485](../../app/websocket/chat.py#L484-L485) and
[:1556-1557](../../app/websocket/chat.py#L1556-L1557)), which read
`service.is_monitoring`. So the `monitoring_active` flag in the
`chat_connected` and `agent_state_changed` payloads is permanently `False`
even while the loop is running.

### 4.4 [Low] Signature-set eviction removes arbitrary entries, not oldest

[monitoring_loop.py:329-334](../../agents/core/monitoring_loop.py#L329-L334)
comments "Evict oldest entries" but iterates a `set`, whose order is by hash
bucket, not insertion. It can evict a board inspected seconds ago, causing a
re-inspection. Use an `OrderedDict`/`deque` if recency matters — or note that
at 100 000 entries the cap is unlikely to be reached and simplify.

### 4.5 [Low] Leftover marker constants at the end of a module

[app/database/connection.py:491-492](../../app/database/connection.py#L491-L492):

```python
CONNECTION_MODULE_READY = True
EOF_MARKER = CONNECTION_MODULE_READY
```

Nothing reads either. Editing artifact; delete.

---

## 5. Dead code still in the active tree

Verified unreferenced but **not** archived, because removing them means
deleting code rather than relocating a file — flagged here for a decision:

| Location | Dead weight | Notes |
|---|---|---|
| `router/resilience.py` | ~700 of 1091 lines | Only `calculate_backoff`, `get_concurrency_limiter`, `generate_request_id`, `RequestMetrics`, `_request_logger` are used (by `router/adapters/vllm.py`). `ResilientLLMClient`, `make_resilient_request`, `with_retry`, `RequestDeduplicator`, `RateLimitException`, `estimate_tokens` have no callers — it is an Anthropic/OpenAI-SDK client sitting in a vLLM-only app. |
| `services/core/monitoring.py:982-1047` | `_monitoring_loop` (~65 lines) | Legacy reactive loop. No thread ever starts it. Its supporting state (`_analysis_executor`, `_analysis_futures`, `analysis_workers`, `max_pending_analyses`, `ssim_threshold`, `motion_burst_interval`, `_last_backpressure_log`) and `_clear_pending_queue` are dead with it. |
| `app/database/repositories.py:701-871` | `cleanup_old`, `deduplicate_similar` (~140 lines) | See 4.2 — wire up or delete. |
| `app/api/v1/logs.py:200-247` | `/api/recent_images` | No template or script calls it. It also `stat()`s every `.jpg` in two directories and inlines 15 base64 images into one JSON response — expensive if it ever *is* called. |

---

## 6. Deployment weight

### 6.1 [Medium] Five unused dependencies, two of them heavy

`requirements/base.txt` declares packages nothing imports (verified by grep
across the tree):

- `anthropic`, `openai`, `google-genai` — cloud LLM SDKs. All inference goes
  through `router/adapters/vllm.py`, which uses plain `requests`.
- `gTTS` — no text-to-speech anywhere.
- `weasyprint` — no PDF generation anywhere. This one also drags
  `libpango-1.0-0`, `libpangocairo-1.0-0`, `libgdk-pixbuf-2.0-0`, `libffi-dev`
  and `shared-mime-info` into the `apt-get install` in [Dockerfile:5-22](../../Dockerfile#L5-L22).

Dropping them shrinks the image and the CVE surface. `alsa-utils`/`ffmpeg` can
go too if audio stays retired (see `archive/README.md`).

### 6.2 [Medium] The production image installs the test toolchain

[Dockerfile:26-31](../../Dockerfile#L26-L31) installs root `requirements.txt`,
which is `-r requirements/prod.txt` **+ `-r requirements/test.txt`** — so
pytest, pytest-cov, pytest-mock, pytest-asyncio and `responses` ship in the
runtime image. `requirements/prod.txt` alone is what the image wants.

### 6.3 [Medium] A 247 MB video sits in the Docker build context

`IMG_1043.MOV` is gitignored but **not** in `.dockerignore`, so every
`docker build` uploads 247 MB of context and bakes it into a layer. The
Dockerfile then `mv`s it into `/app/video/`. Mounting it as a volume (or
pulling it at deploy time) keeps the image small; today the demo asset is
likely the largest single layer.

### 6.4 [Medium] Four CDN dependencies on an edge appliance

[templates/base.html:35-44](../../templates/base.html#L35-L44) loads Tailwind,
Font Awesome, Socket.IO and Chart.js from `cdn.tailwindcss.com`,
`cdnjs.cloudflare.com` and `cdn.jsdelivr.net`. On an air-gapped or
intermittently-connected industrial deployment the UI degrades to unstyled HTML
with no charts and no WebSocket client.

`cdn.tailwindcss.com` specifically is the **browser-side JIT compiler**, which
its own docs mark as development-only: it ships a compiler to every client and
generates CSS at runtime on each page load. Vendor all four into `static/`.

---

## 7. Repository cleanup performed

Moved to `archive/` (paths mirrored, `git mv` so history follows). Full
rationale in [archive/README.md](../../archive/README.md).

| Moved | Size / count | Why |
|---|---|---|
| `config/` package | 4 files | Unused Pydantic settings duplicating `core/config.py`. Nothing imported it. Also still declared `socketio_cors="*"`, contradicting the CORS fix in `core/config.py`. |
| `core/errors.py` | 12 exception classes | Never raised or caught. Re-export dropped from `core/__init__.py`. |
| `services/infrastructure/audio.py` | `AudioPlayer` | Never imported. (Finding 2.4 of the first review fixed a real bug in code no caller reaches.) |
| `app/database/base_repository.py` | `BaseRepository` ABC | No repository subclasses it. Re-export dropped from `app/database/__init__.py`. |
| `helm/*.tgz` + root `.tgz` | 19 archives | `helm package` build output; regenerable, and sitting next to the chart source invited editing a published archive by mistake. |
| `DEBUG.md` | 4.8 KB | One-off Helm debugging notes, already superseded by the chart and the deployment guide. |

Also deleted: `config/.settings.py.swp` (vim swap file), `config/__pycache__/`.
Also added `archive/` to `.dockerignore`.

**Verification after the move:** `pytest tests/ -q` → 38 passed;
`python -m compileall app agents core services router run.py wsgi.py` → clean;
`create_app()` builds all 80 routes; `helm lint helm/camera-agent` → 0 failed.

Left in place deliberately: `IMG_1043.MOV` (live Dockerfile asset),
`camera_agent.log` / `camera_agent.db` (owned by the running `run.py`
process), `agents/mcp/{executor,interpreter,tool_defs}.py` (look like dead
shims, are actually imported), `BLOG.md`, `requirements.txt`.

---

## Suggested order

1. **4.1** — pooled connections without rollback is the one that corrupts
   shared state rather than just wasting cycles.
2. **1.1 + 1.2** — biggest efficiency win, ~40 ms/s of a core reclaimed and one
   source of truth for board presence instead of two that can disagree.
3. **4.2** — unbounded disk growth will eventually take the appliance down.
4. **2** — chat history; cheap fixes, and the degradation is gradual enough to
   go unnoticed until a demo.
5. **6.1–6.3** — dependency and image trimming; mechanical, no behaviour change.
6. **5** — delete the dead modules once 4.2's disposition is decided.
