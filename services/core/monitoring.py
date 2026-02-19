"""Camera monitoring service.

Provides the StreamlinedMonitoringService that subscribes to the
camera feed and runs the unified VLM-based detection pipeline.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from agents.core.detection_agent import StreamlinedAgent
from agents.core.monitoring_loop import MonitoringLoop
from core.resilience import CircuitBreaker
from agents.vlm.task_types import TaskType
from agents.core.state import DetectionEvent
from services.core.camera import get_camera_publisher
from services.infrastructure.vlm import create_vlm_client_from_config
from services.infrastructure.config import load_camera_config
from core.logging import get_logger

logger = get_logger(__name__)


DEFAULT_MONITORING_DEFECT_PROMPT = """You are a PCB quality inspector. Examine this image carefully. Output ONLY valid JSON.

BOARD: Arduino Uno R4 Minima on a green surface, viewed at a slight angle.

IMPORTANT: You MUST look at each connector location individually and describe what you physically see there before deciding its status. Do NOT assume all connectors are present just because this is an Arduino board.

STEP 1 — Look at the LEFT EDGE of the board for the DC Power Barrel Jack:
- A present barrel jack is a LARGE BLACK CYLINDRICAL plastic body (~9mm wide, ~11mm tall) that sticks out from the board edge. It is the TALLEST component on the board edge.
- If you see only flat bare copper pads, solder points, or empty through-holes at that location with NO tall black cylinder, the jack is MISSING.
- Describe what you see at this location in your reasoning.

STEP 2 — Look at the TOP EDGE for the USB connector:
- A small metallic rectangular port. Much smaller and flatter than the barrel jack.
- Describe what you see.

STEP 3 — Check pin headers for protective caps/jumpers.

STEP 4 — General defect scan: solder_bridge, missing_component, cold_solder_joint, lifted_pad, trace_damage, connector_misalignment, mechanical_damage. Report only what has visible evidence.

SEVERITY: high = missing connector or structural damage | medium = cold joint, lifted pad | low = cosmetic only

Return this JSON:
{
    "detected": bool,
    "confidence": float (0-1),
    "reasoning": "Describe what you see at each connector location, then state your conclusion",
    "should_alert": bool,
    "details": {
        "power_jack_status": "present_intact|missing|damaged|uncertain",
        "uart_cap_status": "present_both_sides|missing_one_side|missing_both_sides|uncertain",
        "defects": [
            {
                "type": "defect_type",
                "severity": "low|medium|high|uncertain",
                "location": "where on the board",
                "description": "what you physically see"
            }
        ]
    }
}

detected=true if ANY defect found. should_alert=true for medium/high severity.
"""


class MonitoringMode(str, Enum):
    """Supported monitoring runtime modes."""

    AUTO = "auto"
    PROACTIVE = "proactive"


@dataclass(frozen=True)
class MonitoringModeDecision:
    """Decision payload for monitoring mode selection."""

    mode: MonitoringMode
    reason: str


class MonitoringModeSelector:
    """Deterministic strategy for selecting monitoring mode."""

    @staticmethod
    def choose_mode(  # pylint: disable=too-many-arguments
        requested_mode: MonitoringMode,
        *,
        proactive_config: Dict[str, Any],
        instruction: str,
        task_type: TaskType,
        custom_prompt: str,
        proactive_running: bool,
    ) -> MonitoringModeDecision:
        _ = (requested_mode, proactive_config, instruction, task_type, custom_prompt, proactive_running)
        return MonitoringModeDecision(
            mode=MonitoringMode.PROACTIVE,
            reason="proactive_only_runtime",
        )

    @staticmethod
    def _is_proactive_scope_instruction(instruction: str) -> bool:
        try:
            MonitoringLoop.validate_instruction_scope(instruction)
            return True
        except ValueError:
            return False


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

# Global singleton for monitoring service
_monitoring_service: Optional["StreamlinedMonitoringService"] = None
_service_lock = threading.Lock()


def get_monitoring_service() -> Optional["StreamlinedMonitoringService"]:
    """Get or create the global monitoring service instance."""
    global _monitoring_service
    
    with _service_lock:
        if _monitoring_service is None:
            _monitoring_service = StreamlinedMonitoringService()
        return _monitoring_service


class StreamlinedMonitoringService:
    """
    Camera monitoring controller using the streamlined VLM pipeline.
    """

    def __init__(
        self,
        subscriber_id: Optional[str] = None,
        publisher_getter: Callable = get_camera_publisher,
        auto_start_publisher: bool = True,
    ) -> None:
        self.agent: Optional[StreamlinedAgent] = None
        self.subscriber_id = subscriber_id or f"monitor_{uuid4().hex[:8]}"
        self.is_monitoring = False
        self.last_frame: Optional[Dict[str, object]] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._publisher_getter = publisher_getter
        self.publisher = None
        self.auto_start_publisher = auto_start_publisher
        
        # Dynamic prompt configuration
        self._current_task_type: TaskType = TaskType.CUSTOM
        self._custom_prompt: str = ""
        self._alerts_enabled: bool = False
        self._agentic_mode: bool = False
        self._prompt_version: int = 0
        self._prompt_lock = threading.Lock()
        
        # Inference lock
        self._inference_lock = threading.Lock()
        
        # Timing configuration
        self.last_processed_time = 0.0
        self.capture_interval = 30.0
        self.analysis_workers = 1
        self.max_pending_analyses = 3
        
        # Analysis executor
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        self._analysis_futures = deque()
        self._futures_lock = threading.Lock()
        
        # Proactive agent state
        self.proactive_agent: Optional[MonitoringLoop] = None
        self._proactive_instruction: str = ""
        self._proactive_config: Dict[str, Any] = {}
        self._proactive_lock = threading.RLock()
        self._mode_selector = MonitoringModeSelector()
        self._active_mode = MonitoringMode.PROACTIVE
        self._last_mode_decision_reason = "not_started"

        # Motion detection
        self.ssim_threshold = 0.80
        self.motion_burst_interval = 0.2
        self._last_backpressure_log = 0.0
        
        # Stats tracking
        self._stats_lock = threading.Lock()
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'alerts_sent': 0,
            'dropped_frames': 0,
            'single_frame_requests': 0,
            'analysis_avg_ms': 0.0,
            'analysis_samples': 0,
            'uptime_start': datetime.now().isoformat()
        }

    def _apply_configuration_settings(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings to runtime parameters."""
        camera_cfg = config.get("camera", {})
        advanced_cfg = config.get("advanced", {})
        
        self.capture_interval = float(camera_cfg.get("capture_interval", self.capture_interval))
        self.analysis_workers = int(advanced_cfg.get("max_concurrent_analyses", self.analysis_workers))
        self.max_pending_analyses = int(advanced_cfg.get("max_pending_analyses", self.max_pending_analyses))

    def _ensure_analysis_executor(self) -> None:
        """Ensure the analysis executor is running."""
        if self._analysis_executor is None:
            self._analysis_executor = ThreadPoolExecutor(max_workers=self.analysis_workers)

    def initialize(self) -> bool:
        """Initialize the monitoring service."""
        try:
            config = load_camera_config()
            vlm_client = create_vlm_client_from_config(config)
            circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120.0)
            self._proactive_config = config.get("proactive", {}) or {}
            
            self.agent = StreamlinedAgent(
                config=config,
                vlm_client=vlm_client,
                circuit_breaker=circuit_breaker,
            )
            
            # Initialize publisher
            self.publisher = self._publisher_getter()
            if self.auto_start_publisher and not self.publisher.is_running:
                self.publisher.start()
            
            logger.info("Monitoring service initialized")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error("Failed to initialize monitoring service: %s", e)
            return False

    def start_monitoring(
        self,
        *,
        mode: Optional[MonitoringMode | str] = None,
        instruction: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Start monitoring in proactive mode only."""
        requested_mode = _parse_monitoring_mode(mode)

        if self.is_monitoring:
            self.last_error = "Monitoring already active"
            return False

        # Avoid running both reactive and proactive loops simultaneously
        if self.proactive_agent and self.proactive_agent.is_running:
            self.last_error = "Proactive monitoring is currently active"
            return False
        
        if not self.agent:
            if not self.initialize():
                return False
        agent = self.agent
        if agent is None:
            self.last_error = "Agent initialization failed"
            return False

        if self.publisher is None:
            self.publisher = self._publisher_getter()
        if self.auto_start_publisher and self.publisher and not self.publisher.is_running:
            self.publisher.start()

        proactive_cfg: Dict[str, Any] = {}
        try:
            proactive_cfg = (agent.config or {}).get("proactive", {}) or {}
        except Exception:
            proactive_cfg = {}

        if config:
            proactive_cfg = {**proactive_cfg, **config}

        with self._prompt_lock:
            task_type = self._current_task_type
            custom_prompt = self._custom_prompt

        decision = self._mode_selector.choose_mode(
            requested_mode,
            proactive_config=proactive_cfg,
            instruction=instruction,
            task_type=task_type,
            custom_prompt=custom_prompt,
            proactive_running=bool(
                self.proactive_agent and self.proactive_agent.is_running
            ),
        )
        self._last_mode_decision_reason = decision.reason

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
        self._last_mode_decision_reason = decision.reason
        logger.info(
            "Monitoring started in %s mode (%s)",
            self._active_mode.value,
            decision.reason,
        )
        return True

    def stop_monitoring(self) -> None:
        """Stop monitoring (proactive runtime and any legacy reactive loop)."""
        self.is_monitoring = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        if self.publisher:
            try:
                self.publisher.unsubscribe(self.subscriber_id)
            except Exception:
                pass

        self.stop_proactive_monitoring()
        
        logger.info("Monitoring stopped")

    # ------------------------------------------------------------------
    # Proactive control
    # ------------------------------------------------------------------

    def start_proactive_monitoring(
        self,
        instruction: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Start or update the proactive monitoring agent."""
        if not instruction or not instruction.strip():
            self.last_error = "Instruction is required"
            return False

        # Scope enforcement: proactive mode is intentionally PCB-defect-only.
        # Non-PCB requests are refused explicitly at the service boundary.
        try:
            MonitoringLoop.validate_instruction_scope(instruction)
        except ValueError as exc:
            self.last_error = str(exc)
            return False

        if not self.agent:
            if not self.initialize():
                return False

        agent = self.agent
        if agent is None:
            self.last_error = "Agent initialization failed"
            return False

        if self.is_monitoring:
            logger.info("Pausing reactive monitoring while proactive mode is active")
            self.stop_monitoring()

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
            if not self.is_monitoring:
                self._active_mode = MonitoringMode.PROACTIVE

    def get_active_monitoring_mode(self) -> str:
        """Return the currently active runtime monitoring mode."""
        if self.proactive_agent and self.proactive_agent.is_running:
            return MonitoringMode.PROACTIVE.value
        return "idle"

    def get_proactive_snapshot(self) -> Dict[str, Any]:
        with self._proactive_lock:
            if self.proactive_agent:
                snapshot = self.proactive_agent.snapshot()
            else:
                snapshot = {"running": False, "context": None}
            snapshot["instruction"] = self._proactive_instruction
            return snapshot

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

    def _persist_inspection(
        self, event: DetectionEvent, board_signature: str
    ) -> None:
        """Persist pass/fail as first-class PCB records for analytics."""
        try:
            from app.database import PCBInspectionRepository  # pylint: disable=import-outside-toplevel

            result = "FAIL" if event.detected else "PASS"
            PCBInspectionRepository.create(
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

    def set_active_prompt(
        self,
        task_type: TaskType,
        custom_prompt: str = "",
        alerts_enabled: bool = False,
        agentic_mode: bool = False,
    ) -> None:
        """Set the active task type and prompt for monitoring."""
        with self._prompt_lock:
            prompt_changed = (
                self._current_task_type != task_type or
                self._custom_prompt != custom_prompt
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

    def analyze_single_frame(
        self,
        task_type: Optional[TaskType] = None,
        custom_prompt: Optional[str] = None,
    ) -> Optional[DetectionEvent]:
        """Analyze the latest frame immediately."""
        if not self.agent:
            self.last_error = "Agent not initialized"
            return None
        
        with self._stats_lock:
            self.stats['single_frame_requests'] += 1
        
        # Get latest frame
        publisher = self._publisher_getter()
        latest = publisher.get_latest_frame()
        
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

    def refresh_configuration(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Refresh service configuration.
        
        If no config is provided, loads from agent's config or disk.
        """
        if config is None and self.agent:
            config = self.agent.config
        if config is None:
            config = load_camera_config()
        
        self._apply_configuration_settings(config)
        if self.agent and hasattr(self.agent, "apply_config"):
            self.agent.apply_config(config)

    def _run_analysis(
        self,
        frame: Any,
        task_type: TaskType,
        custom_prompt: str,
        agentic: bool,
    ) -> Optional[DetectionEvent]:
        """Run a single analysis pass (agentic or standard)."""
        agent = self.agent
        if agent is None:
            raise RuntimeError("Agent not initialized")

        effective_prompt = (
            custom_prompt.strip()
            if isinstance(custom_prompt, str) and custom_prompt.strip()
            else DEFAULT_MONITORING_DEFECT_PROMPT
        )

        if agentic:
            config = load_camera_config()
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

    def _serialize_stats(self) -> Dict[str, Any]:
        """Get a copy of current stats."""
        with self._stats_lock:
            payload = self.stats.copy()
        payload["proactive"] = self.get_proactive_snapshot()
        payload["monitoring_mode"] = self.get_active_monitoring_mode()
        payload["mode_decision_reason"] = self._last_mode_decision_reason
        return payload

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        publisher = self.publisher or self._publisher_getter()
        self.publisher = publisher

        if not publisher:
            logger.error("Monitoring loop has no publisher")
            self.is_monitoring = False
            return
        
        try:
            while self.is_monitoring:
                try:
                    frame_obj = publisher.get_frame(self.subscriber_id, timeout=1.0)
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
                publisher.unsubscribe(self.subscriber_id)
            except Exception:
                pass

    def _record_detection(self, event: DetectionEvent, metadata: Dict[str, Any]) -> None:
        """Record detection to database and emit websocket event."""
        from app.database import DetectionLogRepository
        from app import socketio
        
        try:
            log_id = DetectionLogRepository.create(
                timestamp=event.timestamp,
                confidence=event.confidence,
                response=event.full_response,
                image_path=event.image_path or "",
                frame_number=metadata.get("frame_number"),
                reason=str(metadata.get("reason") or ""),
                vision_description=getattr(event, "vision_description", ""),
                decision_details=event.decision_trace,
                tool_trace=getattr(event, "tool_trace", []),
            )
            
            # Emit real-time event
            if log_id:
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
                socketio.emit("new_log", log_entry)
                
        except Exception as exc:
            logger.error("Failed to record detection: %s", exc)
