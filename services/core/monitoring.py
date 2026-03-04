"""Camera monitoring service.

Provides the StreamlinedMonitoringService that subscribes to the
camera feed and runs the unified VLM-based detection pipeline.

Design notes
~~~~~~~~~~~~
* **Proactive-only runtime** — The only active monitoring mode is
  ``PROACTIVE``.  The ``MonitoringMode`` enum retains an ``IDLE``
  member for API consumers that inspect the current mode; there is no
  separate "reactive" runtime.
* **Dependency injection** — Database and SocketIO helpers are injected
  at construction time (``detection_repo``, ``inspection_repo``,
  ``socketio_emitter``) so the service is testable in isolation and
  never performs deferred imports from ``app.*`` in background threads.
  For backward-compatibility the service *also* supports late-import
  resolution when no dependency was injected, caching the reference on
  first successful import.
* **Thread safety** — ``is_monitoring``, ``agent``, and ``publisher``
  are guarded by ``_core_lock``.  Stats and prompt state each have
  their own dedicated locks, which are never held simultaneously.
* **Config caching** — ``load_camera_config()`` is called in
  ``initialize()`` and cached; ``_run_analysis`` reads from the cache.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol
from uuid import uuid4

from agents.core.detection_agent import StreamlinedAgent
from agents.core.monitoring_loop import MonitoringLoop
from core.resilience import CircuitBreaker
from agents.vlm.task_types import TaskType
from agents.vlm.prompts import DEFAULT_MONITORING_DEFECT_PROMPT
from agents.core.state import DetectionEvent
from services.core.camera import get_camera_publisher
from services.infrastructure.vlm import create_vlm_client_from_config
from services.infrastructure.config import load_camera_config
from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Monitoring mode
# ---------------------------------------------------------------------------

class MonitoringMode(str, Enum):
    """Supported monitoring runtime modes."""

    IDLE = "idle"
    PROACTIVE = "proactive"


# ---------------------------------------------------------------------------
# Dependency protocols (for injection)
# ---------------------------------------------------------------------------

class DetectionLogRepo(Protocol):
    """Minimal interface for the detection-log persistence layer."""

    @staticmethod
    def create(
        *,
        timestamp: str,
        confidence: float,
        response: str,
        image_path: str,
        frame_number: Any,
        reason: str,
        vision_description: str,
        decision_details: Any,
        tool_trace: Any,
    ) -> Optional[int]: ...


class PCBInspectionRepo(Protocol):
    """Minimal interface for the PCB-inspection persistence layer."""

    @staticmethod
    def create(
        *,
        board_signature: str,
        result: str,
        confidence: float,
        reason: str,
        defect_type: str,
        description: str,
        image_path: str,
        decision_trace: dict,
    ) -> Optional[int]: ...


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_monitoring_mode(
    mode: Optional[MonitoringMode | str],
) -> MonitoringMode:
    if isinstance(mode, MonitoringMode):
        return mode
    if isinstance(mode, str) and mode.strip():
        normalized = mode.strip().lower()
        for candidate in MonitoringMode:
            if candidate.value == normalized:
                return candidate
    return MonitoringMode.PROACTIVE


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_monitoring_service: Optional["StreamlinedMonitoringService"] = None
_service_lock = threading.Lock()


def get_monitoring_service() -> Optional["StreamlinedMonitoringService"]:
    """Get or create the global monitoring service instance.

    The returned service may not be *initialized* yet (i.e. the VLM
    client and camera publisher are not connected).  Call
    ``service.is_ready`` to check, or ``service.initialize()`` to
    perform first-time setup.  Most public methods that need the agent
    will call ``initialize()`` lazily if it has not been done.
    """
    global _monitoring_service

    with _service_lock:
        if _monitoring_service is None:
            _monitoring_service = StreamlinedMonitoringService()
        return _monitoring_service


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class StreamlinedMonitoringService:
    """Camera monitoring controller using the streamlined VLM pipeline.

    Parameters
    ----------
    subscriber_id : str, optional
        Unique ID used when subscribing to the camera publisher.
    publisher_getter : callable
        Factory that returns a camera publisher instance.
    auto_start_publisher : bool
        When *True* the publisher is started automatically on first use.
    detection_repo : object, optional
        Injected ``DetectionLogRepository``-like object.  When *None*
        the service will fall back to a lazy import from
        ``app.database`` (kept for backward-compat; prefer explicit
        injection in new code and tests).
    inspection_repo : object, optional
        Injected ``PCBInspectionRepository``-like object.
    socketio_emitter : callable, optional
        ``socketio.emit``-compatible callable.  When *None* the
        service will fall back to a lazy import from ``app``.
    """

    def __init__(
        self,
        subscriber_id: Optional[str] = None,
        publisher_getter: Callable = get_camera_publisher,
        auto_start_publisher: bool = True,
        *,
        detection_repo: Optional[DetectionLogRepo] = None,
        inspection_repo: Optional[PCBInspectionRepo] = None,
        socketio_emitter: Optional[Callable] = None,
    ) -> None:
        # --- Core state (guarded by _core_lock) -----------------------
        self._core_lock = threading.Lock()
        self._agent: Optional[StreamlinedAgent] = None
        self._publisher: Optional[Any] = None
        self._is_monitoring: bool = False
        self._initialized: bool = False

        self.subscriber_id = subscriber_id or f"monitor_{uuid4().hex[:8]}"
        self.last_frame: Optional[Dict[str, object]] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._publisher_getter = publisher_getter
        self.auto_start_publisher = auto_start_publisher

        # --- Injected dependencies ------------------------------------
        self._detection_repo = detection_repo
        self._inspection_repo = inspection_repo
        self._socketio_emit = socketio_emitter

        # --- Dynamic prompt configuration (guarded by _prompt_lock) ---
        self._current_task_type: TaskType = TaskType.CUSTOM
        self._custom_prompt: Optional[str] = None
        self._alerts_enabled: bool = False
        self._agentic_mode: bool = False
        self._prompt_version: int = 0
        self._prompt_lock = threading.Lock()

        # --- Inference lock -------------------------------------------
        self._inference_lock = threading.Lock()

        # --- Timing configuration -------------------------------------
        self.last_processed_time = 0.0
        self.capture_interval = 30.0
        self.analysis_workers = 1
        self.max_pending_analyses = 3

        # --- Analysis executor ----------------------------------------
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_futures: deque = deque()
        self._futures_lock = threading.Lock()

        # --- Proactive agent state (guarded by _proactive_lock) -------
        self.proactive_agent: Optional[MonitoringLoop] = None
        self._proactive_instruction: str = ""
        self._proactive_config: Dict[str, Any] = {}
        self._proactive_lock = threading.RLock()
        self._active_mode: MonitoringMode = MonitoringMode.IDLE
        self._last_mode_decision_reason = "not_started"

        # --- Cached config (set during initialize) --------------------
        self._cached_config: Dict[str, Any] = {}

        # --- Motion detection -----------------------------------------
        self.ssim_threshold = 0.80
        self.motion_burst_interval = 0.2
        self._last_backpressure_log = 0.0

        # --- Stats tracking (guarded by _stats_lock) ------------------
        self._stats_lock = threading.Lock()
        self.stats: Dict[str, Any] = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'dropped_frames': 0,
            'single_frame_requests': 0,
            'analysis_avg_ms': 0.0,
            'analysis_samples': 0,
            'uptime_start': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Properties (thread-safe accessors for core state)
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return *True* if ``initialize()`` has completed successfully."""
        with self._core_lock:
            return self._initialized

    @property
    def agent(self) -> Optional[StreamlinedAgent]:
        with self._core_lock:
            return self._agent

    @agent.setter
    def agent(self, value: Optional[StreamlinedAgent]) -> None:
        with self._core_lock:
            self._agent = value

    @property
    def publisher(self) -> Optional[Any]:
        with self._core_lock:
            return self._publisher

    @publisher.setter
    def publisher(self, value: Optional[Any]) -> None:
        with self._core_lock:
            self._publisher = value

    @property
    def is_monitoring(self) -> bool:
        with self._core_lock:
            return self._is_monitoring

    @is_monitoring.setter
    def is_monitoring(self, value: bool) -> None:
        with self._core_lock:
            self._is_monitoring = value

    # ------------------------------------------------------------------
    # Dependency resolution helpers
    # ------------------------------------------------------------------

    def _get_detection_repo(self) -> Any:
        """Return the detection-log repository (injected or late-imported)."""
        if self._detection_repo is not None:
            return self._detection_repo
        try:
            from app.database import DetectionLogRepository  # pylint: disable=import-outside-toplevel
            self._detection_repo = DetectionLogRepository
            return DetectionLogRepository
        except Exception:
            return None

    def _get_inspection_repo(self) -> Any:
        """Return the PCB-inspection repository (injected or late-imported)."""
        if self._inspection_repo is not None:
            return self._inspection_repo
        try:
            from app.database import PCBInspectionRepository  # pylint: disable=import-outside-toplevel
            self._inspection_repo = PCBInspectionRepository
            return PCBInspectionRepository
        except Exception:
            return None

    def _get_socketio_emit(self) -> Optional[Callable]:
        """Return a ``socketio.emit`` callable (injected or late-imported)."""
        if self._socketio_emit is not None:
            return self._socketio_emit
        try:
            from app import socketio  # pylint: disable=import-outside-toplevel
            self._socketio_emit = socketio.emit
            return socketio.emit
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _apply_configuration_settings(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings to runtime parameters."""
        camera_cfg = config.get("camera", {})
        advanced_cfg = config.get("advanced", {})

        self.capture_interval = float(camera_cfg.get("capture_interval", self.capture_interval))
        self.analysis_workers = int(advanced_cfg.get("max_concurrent_analyses", self.analysis_workers))
        self.max_pending_analyses = int(advanced_cfg.get("max_pending_analyses", self.max_pending_analyses))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the monitoring service (VLM client, publisher, config).

        Safe to call multiple times; subsequent calls are no-ops if the
        service is already initialized.
        """
        with self._core_lock:
            if self._initialized:
                return True

        try:
            config = load_camera_config()
            self._cached_config = config
            vlm_client = create_vlm_client_from_config(config)
            circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120.0)
            self._proactive_config = config.get("proactive", {}) or {}

            agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                circuit_breaker=circuit_breaker,
            )

            pub = self._publisher_getter()
            if self.auto_start_publisher and not pub.is_running:
                pub.start()

            with self._core_lock:
                self._agent = agent
                self._publisher = pub
                self._initialized = True

            self._apply_configuration_settings(config)
            logger.info("Monitoring service initialized")
            return True

        except Exception as e:
            self.last_error = str(e)
            logger.error("Failed to initialize monitoring service: %s", e)
            return False

    def _ensure_initialized(self) -> bool:
        """Lazily initialize if needed.  Returns *True* on success."""
        if self.is_ready:
            return True
        return self.initialize()

    # ------------------------------------------------------------------
    # Start / stop (public API)
    # ------------------------------------------------------------------

    def start_monitoring(
        self,
        *,
        mode: Optional[MonitoringMode | str] = None,
        instruction: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Start proactive monitoring.

        This is the single entry-point for callers that want to begin
        monitoring.  The ``mode`` parameter is accepted for API
        compatibility but the runtime is always proactive.
        """
        _parse_monitoring_mode(mode)  # validates; currently always proactive

        if self.is_monitoring:
            self.last_error = "Monitoring already active"
            return False

        # Avoid running both reactive and proactive loops simultaneously
        if self.proactive_agent and self.proactive_agent.is_running:
            self.last_error = "Proactive monitoring is currently active"
            return False

        if not self._ensure_initialized():
            return False

        agent = self.agent
        if agent is None:
            self.last_error = "Agent initialization failed"
            return False

        # Ensure publisher is alive
        pub = self.publisher
        if pub is None:
            pub = self._publisher_getter()
            self.publisher = pub
        if self.auto_start_publisher and pub and not pub.is_running:
            pub.start()

        proactive_cfg: Dict[str, Any] = {}
        try:
            proactive_cfg = (agent.config or {}).get("proactive", {}) or {}
        except Exception:
            proactive_cfg = {}

        if config:
            proactive_cfg = {**proactive_cfg, **config}

        chosen_instruction = (
            (instruction or "").strip()
            or str(
                proactive_cfg.get(
                    "default_instruction",
                    "Inspect moving PCBs on the conveyor and flag defects.",
                )
            ).strip()
        )

        started = self.start_proactive_monitoring(
            instruction=chosen_instruction,
            config=proactive_cfg,
        )
        if not started:
            return False

        self._active_mode = MonitoringMode.PROACTIVE
        self._last_mode_decision_reason = "proactive_only_runtime"
        logger.info(
            "Monitoring started in %s mode (%s)",
            self._active_mode.value,
            self._last_mode_decision_reason,
        )
        return True

    def stop_monitoring(self) -> None:
        """Stop monitoring (proactive runtime and any legacy reactive loop)."""
        self.is_monitoring = False

        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

        pub = self.publisher
        if pub:
            try:
                pub.unsubscribe(self.subscriber_id)
            except Exception:
                pass

        self.stop_proactive_monitoring()
        self._active_mode = MonitoringMode.IDLE
        logger.info("Monitoring stopped")

    # ------------------------------------------------------------------
    # Proactive control
    # ------------------------------------------------------------------

    def start_proactive_monitoring(
        self,
        instruction: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Start or update the proactive monitoring agent.

        If proactive monitoring is already running, the instruction is
        updated in-place without restarting the loop.  If a legacy
        reactive loop is active it will be stopped cleanly first
        *without* tearing down the proactive agent.
        """
        if not instruction or not instruction.strip():
            self.last_error = "Instruction is required"
            return False

        # Scope enforcement: proactive mode is intentionally PCB-defect-only.
        try:
            MonitoringLoop.validate_instruction_scope(instruction)
        except ValueError as exc:
            self.last_error = str(exc)
            return False

        if not self._ensure_initialized():
            return False

        agent = self.agent
        if agent is None:
            self.last_error = "Agent initialization failed"
            return False

        # Stop only the legacy reactive loop if it is running,
        # WITHOUT touching the proactive agent (avoids circular teardown
        # that previously occurred when stop_monitoring() called
        # stop_proactive_monitoring()).
        if self.is_monitoring:
            logger.info("Stopping reactive loop before starting proactive mode")
            self.is_monitoring = False
            thread = self._thread
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            pub = self.publisher
            if pub:
                try:
                    pub.unsubscribe(self.subscriber_id)
                except Exception:
                    pass

        with self._proactive_lock:
            merged_config = {**self._proactive_config, **(config or {})}
            self._proactive_config = merged_config

            if self.proactive_agent and self.proactive_agent.is_running:
                try:
                    self.proactive_agent.update_instruction(instruction)
                except ValueError as exc:
                    self.last_error = str(exc)
                    return False
                self._proactive_instruction = instruction.strip()
                return True

            self.proactive_agent = MonitoringLoop(
                instruction=instruction,
                publisher_getter=self._publisher_getter,
                on_board_ready=self._on_board_ready,
                event_callback=self._record_detection,
                config=merged_config,
            )
            self._proactive_instruction = instruction.strip()
            self.proactive_agent.start()
            self._active_mode = MonitoringMode.PROACTIVE
            return True

    def stop_proactive_monitoring(self) -> None:
        """Stop the proactive monitoring agent if running."""
        with self._proactive_lock:
            agent = self.proactive_agent
            self.proactive_agent = None
            if agent and agent.is_running:
                agent.stop()
            self._proactive_instruction = ""

    def get_active_monitoring_mode(self) -> str:
        """Return the currently active runtime monitoring mode.

        Returns a ``MonitoringMode`` *value* string (``"proactive"``
        or ``"idle"``).
        """
        if self.proactive_agent and self.proactive_agent.is_running:
            return MonitoringMode.PROACTIVE.value
        return MonitoringMode.IDLE.value

    def get_proactive_snapshot(self) -> Dict[str, Any]:
        with self._proactive_lock:
            if self.proactive_agent:
                snapshot = self.proactive_agent.snapshot()
            else:
                snapshot = {"running": False, "context": None}
            snapshot["instruction"] = self._proactive_instruction
            return snapshot

    # ------------------------------------------------------------------
    # Board-ready callback (called from MonitoringLoop thread)
    # ------------------------------------------------------------------

    def _on_board_ready(
        self, frame: Any, observation_context: Dict[str, Any]
    ) -> None:
        """Called by MonitoringLoop when a board stops in the zone.

        This is where the LLM decides what to do.  The VLM analyzes the
        frame; the result is recorded so the chat LLM can decide
        follow-up actions (alert, log, etc.) via MCP tools.
        """
        agent = self.agent
        if agent is None:
            logger.warning("on_board_ready: agent not initialized")
            return

        with self._prompt_lock:
            task_type = self._current_task_type
            custom_prompt = self._custom_prompt
            use_agentic = self._agentic_mode

        try:
            event = self._run_analysis(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt,
                agentic=use_agentic,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("on_board_ready analysis failed: %s", exc)
            return

        if event is None:
            return

        # Update stats counters for proactive detections
        with self._stats_lock:
            self.stats['processed_frames'] += 1
            if event.detected:
                self.stats['detections'] += 1

        # Enrich with observation context
        event.decision_trace.setdefault("proactive_context", {})
        event.decision_trace["proactive_context"].update(observation_context)

        # Persist inspection result
        board_sig = observation_context.get("observation", {}).get(
            "board_signature", ""
        )
        self._persist_inspection(event, board_sig)

        self._record_detection(
            event,
            {
                "reason": "proactive_board_ready",
                "board_signature": board_sig,
                **observation_context,
            },
        )

        # Update monitoring loop context
        if self.proactive_agent:
            ctx = self.proactive_agent.context
            ctx.inspections_completed += 1
            if event.detected:
                ctx.defect_found_count += 1
                ctx.board_decisions[board_sig] = "DEFECT_FOUND"
            else:
                ctx.no_defect_count += 1
                ctx.board_decisions[board_sig] = "NO_DEFECT"

    # ------------------------------------------------------------------
    # Persistence helpers (use injected repos, fall back to lazy import)
    # ------------------------------------------------------------------

    def _persist_inspection(
        self, event: DetectionEvent, board_signature: str
    ) -> None:
        """Persist pass/fail as first-class PCB records for analytics."""
        repo = self._get_inspection_repo()
        if repo is None:
            logger.warning("PCBInspectionRepository unavailable; skipping persist")
            return
        try:
            result = "FAIL" if event.detected else "PASS"
            repo.create(
                board_signature=board_signature,
                result=result,
                confidence=float(event.confidence),
                reason=event.vision_description or "",
                defect_type="defect_found" if event.detected else "no_defect",
                description=event.vision_description or event.full_response or "",
                image_path=str(event.image_path or ""),
                decision_trace=dict(event.decision_trace or {}),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Failed to persist inspection: %s", exc)

    def _record_detection(self, event: DetectionEvent, metadata: Dict[str, Any]) -> None:
        """Record detection to database and emit websocket event."""
        repo = self._get_detection_repo()
        emit_fn = self._get_socketio_emit()

        if repo is None:
            logger.warning("DetectionLogRepository unavailable; skipping record")
            return

        try:
            # Enrich decision_details with token usage and tool sequence
            decision_details = dict(event.decision_trace or {})
            token_usage = getattr(event, "token_usage", {})
            if token_usage:
                decision_details["token_usage"] = token_usage
            tools_used = getattr(event, "tools_used", [])
            if tools_used:
                decision_details["tools_used"] = tools_used

            log_id = repo.create(
                timestamp=event.timestamp,
                confidence=event.confidence,
                response=event.full_response,
                image_path=event.image_path or "",
                frame_number=metadata.get("frame_number"),
                reason=str(metadata.get("reason") or ""),
                vision_description=getattr(event, "vision_description", ""),
                decision_details=decision_details,
                tool_trace=getattr(event, "tool_trace", []),
            )

            # Emit real-time events via SocketIO
            if log_id and emit_fn:
                log_entry = {
                    "id": log_id,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "response": event.full_response,
                    "image_path": event.image_path or "",
                    "frame_number": metadata.get("frame_number"),
                    "detected": getattr(event, "detected", False),
                    "agentic_mode": (event.decision_trace or {}).get("classification") == "AGENTIC_ANALYSIS",
                }
                emit_fn("new_log", log_entry)

                # Emit detection_event with current stats for dashboard/chat
                detection_payload = {
                    "event": log_entry,
                    "stats": self._serialize_stats(),
                }
                emit_fn("detection_event", detection_payload)

        except Exception as exc:
            logger.error("Failed to record detection: %s", exc)

    # ------------------------------------------------------------------
    # Prompt configuration
    # ------------------------------------------------------------------

    def set_active_prompt(
        self,
        task_type: TaskType,
        custom_prompt: Optional[str] = None,
        alerts_enabled: bool = False,
        agentic_mode: bool = False,
    ) -> None:
        """Set the active task type and prompt for monitoring.

        ``custom_prompt`` semantics:
        * ``None`` (default) — use the built-in default prompt.
        * Non-empty string — use it as the custom prompt.
        * ``""`` (empty string) — treated the same as ``None``.
        """
        with self._prompt_lock:
            prompt_changed = (
                self._current_task_type != task_type
                or self._custom_prompt != custom_prompt
            )

            if prompt_changed:
                self._prompt_version += 1
                logger.info(
                    "Prompt changed - incrementing version to %d",
                    self._prompt_version
                )
                self._clear_pending_queue()

            self._current_task_type = task_type
            self._custom_prompt = custom_prompt
            self._alerts_enabled = alerts_enabled
            self._agentic_mode = agentic_mode

        logger.info(
            "Active prompt updated: task=%s, alerts=%s, agentic=%s",
            task_type.value,
            alerts_enabled,
            agentic_mode,
        )

    def _clear_pending_queue(self) -> None:
        """Clear all pending analysis futures."""
        cancelled_count = 0
        with self._futures_lock:
            while self._analysis_futures:
                future = self._analysis_futures.popleft()
                if not future.done():
                    future.cancel()
                    cancelled_count += 1

        if cancelled_count > 0:
            logger.info("Cleared %d pending analyses", cancelled_count)

    def get_active_prompt_config(self) -> Dict[str, Any]:
        """Get the current prompt configuration."""
        with self._prompt_lock:
            return {
                "task_type": self._current_task_type.value,
                "custom_prompt": self._custom_prompt,
                "alerts_enabled": self._alerts_enabled,
                "agentic_mode": self._agentic_mode,
                "prompt_version": self._prompt_version,
            }

    # ------------------------------------------------------------------
    # Single-frame analysis
    # ------------------------------------------------------------------

    def analyze_single_frame(
        self,
        task_type: Optional[TaskType] = None,
        custom_prompt: Optional[str] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze the latest frame immediately."""
        if not self._ensure_initialized():
            self.last_error = "Agent not initialized"
            return None

        with self._stats_lock:
            self.stats['single_frame_requests'] += 1

        # Use self.publisher (consistent with rest of service); auto-start
        pub = self.publisher
        if pub is None:
            pub = self._publisher_getter()
            self.publisher = pub
        if self.auto_start_publisher and not pub.is_running:
            pub.start()

        latest = pub.get_latest_frame()

        if not latest or latest.raw_frame is None:
            self.last_error = "No frames available"
            return None

        # Use provided or active config
        with self._prompt_lock:
            effective_task = task_type or self._current_task_type
            effective_prompt = custom_prompt if custom_prompt is not None else self._custom_prompt
            use_agentic = self._agentic_mode

        try:
            event = self._run_analysis(
                frame=latest.raw_frame,
                task_type=effective_task,
                custom_prompt=effective_prompt,
                agentic=use_agentic,
            )

            if event:
                with self._stats_lock:
                    self.stats['processed_frames'] += 1
                    if event.detected:
                        self.stats['detections'] += 1

                self._record_detection(
                    event,
                    {
                        'frame_number': latest.frame_number,
                        'reason': 'single_frame_analysis',
                        'task_type': effective_task.value,
                        'custom_prompt': effective_prompt,
                    },
                )

            return event
        except Exception as e:
            self.last_error = str(e)
            logger.error("Single frame analysis failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Configuration refresh
    # ------------------------------------------------------------------

    def refresh_configuration(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Refresh service configuration.

        If no config is provided, loads from agent's config or disk.
        Also refreshes the cached config used by ``_run_analysis``.
        """
        if config is None and self.agent:
            config = self.agent.config
        if config is None:
            config = load_camera_config()

        self._cached_config = config
        self._apply_configuration_settings(config)
        if self.agent and hasattr(self.agent, "apply_config"):
            self.agent.apply_config(config)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _run_analysis(
        self,
        frame: Any,
        task_type: TaskType,
        custom_prompt: Optional[str],
        agentic: bool,
    ) -> Optional[DetectionEvent]:
        """Run a single analysis pass (agentic or standard).

        ``custom_prompt`` semantics:
        * ``None`` → use the built-in ``DEFAULT_MONITORING_DEFECT_PROMPT``.
        * Non-empty string → use it as-is.
        * Empty string ``""`` → treated as *None* (fall back to default).
        """
        agent = self.agent
        if agent is None:
            raise RuntimeError("Agent not initialized")

        effective_prompt: str
        if isinstance(custom_prompt, str) and custom_prompt.strip():
            effective_prompt = custom_prompt.strip()
        else:
            effective_prompt = DEFAULT_MONITORING_DEFECT_PROMPT

        if agentic:
            # Use cached config instead of re-reading from disk every call
            config = self._cached_config or {}
            recipients = (
                config.get("notifications", {})
                .get("email", {})
                .get("recipients", [])
            )
            return agent.analyze_agentic(
                frame=frame,
                task_type=task_type,
                custom_prompt=effective_prompt,
                recipients=recipients,
            )
        return agent.analyze_with_prompt(
            frame=frame,
            task_type=task_type,
            custom_prompt=effective_prompt,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _serialize_stats(self) -> Dict[str, Any]:
        """Get a copy of current stats."""
        with self._stats_lock:
            payload = self.stats.copy()

        # Sync frame counts from the proactive monitoring loop so the
        # dashboard / chat stats widgets reflect actual activity.
        with self._proactive_lock:
            if self.proactive_agent and hasattr(self.proactive_agent, 'context'):
                ctx = self.proactive_agent.context
                # Use the larger of in-memory counter vs proactive loop counter
                payload['total_frames'] = max(
                    payload['total_frames'],
                    getattr(ctx, 'frames_processed', 0),
                )

        payload["proactive"] = self.get_proactive_snapshot()
        payload["monitoring_mode"] = self.get_active_monitoring_mode()
        payload["mode_decision_reason"] = self._last_mode_decision_reason
        return payload

    # ------------------------------------------------------------------
    # Legacy reactive monitoring loop (retained for backward compat)
    # ------------------------------------------------------------------

    def _monitoring_loop(self) -> None:
        """Main monitoring loop (legacy reactive mode)."""
        pub = self.publisher or self._publisher_getter()
        self.publisher = pub

        if not pub:
            logger.error("Monitoring loop has no publisher")
            self.is_monitoring = False
            return

        try:
            while self.is_monitoring:
                try:
                    frame_obj = pub.get_frame(self.subscriber_id, timeout=1.0)
                    if not frame_obj:
                        continue

                    with self._stats_lock:
                        self.stats['total_frames'] += 1

                    # Check if enough time has passed
                    current_time = time.time()
                    if current_time - self.last_processed_time < self.capture_interval:
                        continue

                    self.last_processed_time = current_time

                    # Get current prompt config
                    with self._prompt_lock:
                        task_type = self._current_task_type
                        custom_prompt = self._custom_prompt
                        agentic = self._agentic_mode

                    # Run analysis
                    try:
                        event = self._run_analysis(
                            frame=frame_obj.raw_frame,
                            task_type=task_type,
                            custom_prompt=custom_prompt,
                            agentic=agentic,
                        )
                    except Exception as e:
                        logger.error("Analysis error: %s", e)
                        continue

                    if event:
                        with self._stats_lock:
                            self.stats['processed_frames'] += 1
                            if event.detected:
                                self.stats['detections'] += 1

                        self._record_detection(event, {
                            'frame_number': frame_obj.frame_number,
                            'reason': 'scheduled_analysis',
                        })

                except Exception as e:
                    logger.error("Monitoring loop error: %s", e)
                    time.sleep(1.0)

        finally:
            try:
                pub.unsubscribe(self.subscriber_id)
            except Exception:
                pass
