# Archive

Files retired from the active tree on 2026-08-06. Nothing here is imported,
executed, or referenced by the running application — each entry was verified
unreferenced before being moved. Paths mirror their original location so any
item can be restored with a straight `git mv` back.

`archive/` is excluded from the Docker build context (`.dockerignore`).

## `config/` — unused Pydantic settings package

`config/{__init__,settings,schemas}.py` + `config/defaults.yaml`.

The live configuration module is `core/config.py` (plain `@dataclass`). This
package duplicated it with Pydantic models and **nothing imported it** —
`grep -rn "from config" --include=*.py` returned zero hits outside the package
itself. `docs/reviews/2026-08-06-documentation-drift.md` flagged the same thing
from the docs side (`docs/deployment/guide.md` still points readers at
`config/settings.py`).

Actively misleading rather than merely dead: `settings.py` still carried
`socketio_cors: str = Field(default="*")`, contradicting the CORS hardening
applied to `core/config.py` in the 2026-08-06 fix pass.

A stray `config/.settings.py.swp` (vim swap file) and `config/__pycache__/`
were deleted rather than archived.

## `core/errors.py` — unused exception hierarchy

Twelve exception classes (`CameraAgentError` and subclasses). Exported from
`core/__init__.py` but never raised or caught anywhere; the codebase uses
built-in exceptions and broad `except Exception` throughout. The re-export was
removed from `core/__init__.py`.

## `services/infrastructure/audio.py` — unused `AudioPlayer`

Platform-agnostic audio playback via system CLI tools (mpv/ffmpeg/aplay/afplay).
Never imported. Not in `services/infrastructure/__init__.py`'s exports either.

Worth noting: finding 2.4 of the 2026-08-06 code review ("`AudioPlayer.is_available()`
is broken") was fixed in code that no caller reaches. If audio alerts are a
planned feature, restore this file; otherwise the Dockerfile's `alsa-utils`
dependency can go too.

## `app/database/base_repository.py` — unused `BaseRepository` ABC

Abstract CRUD base class. No repository in `repositories.py` subclasses it —
they are all plain classes with `@staticmethod` members. The re-export was
removed from `app/database/__init__.py`.

## `helm-releases/` — packaged chart archives

19 `helm package` outputs: 18 that lived in `helm/` alongside the chart source
plus the root-level `zededa-reference-agent-pcb-thor-vllm-2.15.0.tgz`.

These are build artifacts, regenerable from `helm/camera-agent/` with
`helm package`. Keeping them next to the chart source made it easy to edit a
published archive by mistake. They remain tracked here; a follow-up could drop
them from version control entirely and publish to a chart repository instead.

## `notes/DEBUG.md`

Working notes from one specific Helm debugging session (adding NGC/HF keys,
fixing a pending pod). The outcome already landed in `helm/camera-agent/` and
in `docs/deployment/guide.md`. Was gitignored, so it was never tracked.

## Deliberately *not* archived

| Item | Why it stays |
|---|---|
| `IMG_1043.MOV` (247 MB) | Live asset — the Dockerfile stages it as the video simulator (`CAMERA_VIDEO_SOURCE`). Belongs in the tree, but see the review notes about excluding it from the build context. |
| `camera_agent.log`, `camera_agent.db` | Runtime state of the currently-running `run.py` process. Gitignored; leave to the process that owns them. |
| `agents/mcp/{executor,interpreter,tool_defs}.py` | Look like dead re-export shims but are live: `agents/mcp/base.py` imports the first two, `detection_agent.py` and `app/api/v1/analysis.py` import the third. |
| `tests/{e2e,integration}/` | Empty scaffolding (`__init__.py` only), harmless and conventional. |
| `requirements.txt` (root) | Referenced by the Dockerfile. |
| `BLOG.md` | Authored content, not a code artifact — left for its author to place. |
