"""Deterministic proactive PCB monitoring with conveyor FSM enforcement."""
# pylint: disable=no-member
# cv2 attributes are dynamically generated and not visible to pylint

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from core.logging import get_logger
from agents.classifiers import get_classifier
from agents.core.detection_agent import StreamlinedAgent
from agents.core.conveyor_inspection_fsm import (
    ConveyorDecision,
    ConveyorDecisionPayload,
    ConveyorInspectionFSM,
    ConveyorState,
)
from agents.core.state import DetectionEvent
from agents.vlm.client import UnifiedVLMClient
from agents.vlm.task_types import TaskType

logger = get_logger(__name__)


@dataclass
class ObservationResult:  # pylint: disable=too-many-instance-attributes
    """Structured output from the observation step."""

    frame_number: int
    timestamp: str
    scene_summary: str
    target_present: bool
    target_state: str
    confidence: float
    motion_moving: bool
    board_in_zone: bool
    board_signature: Optional[str]
    raw_response: str

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Convert to prompt-friendly payload."""
        return {
            "scene_summary": self.scene_summary,
            "target_present": self.target_present,
            "target_state": self.target_state,
            "confidence": round(self.confidence, 3),
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "motion_moving": self.motion_moving,
            "board_in_zone": self.board_in_zone,
            "board_signature": self.board_signature,
        }


@dataclass
class MonitoringContext:  # pylint: disable=too-many-instance-attributes
    """Runtime context for observability and metrics."""

    instruction: str
    frames_processed: int = 0
    inspections_started: int = 0
    inspections_completed: int = 0
    defect_found_count: int = 0
    no_defect_count: int = 0
    last_observation: Optional[ObservationResult] = None
    last_action_time: float = 0.0
    last_decision: Optional[str] = None
    last_decision_reason: Optional[str] = None
    board_decisions: Dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the monitoring context."""
        return {
            "instruction": self.instruction,
            "frames_processed": self.frames_processed,
            "inspections_started": self.inspections_started,
            "inspections_completed": self.inspections_completed,
            "defect_found_count": self.defect_found_count,
            "no_defect_count": self.no_defect_count,
            "last_observation": (
                self.last_observation.to_prompt_payload()
                if self.last_observation else None
            ),
            "last_action_time": self.last_action_time,
            "last_decision": self.last_decision,
            "last_decision_reason": self.last_decision_reason,
            "board_decisions": self.board_decisions,
        }


class ProactiveMonitoringAgent:  # pylint: disable=too-many-instance-attributes
    """Conveyor-aware proactive monitoring with deterministic FSM control."""

    SCOPE_REFUSAL_MESSAGE = (
        "Refused: proactive monitoring accepts only PCB manufacturing "
        "defect inspection instructions."
    )

    DEFAULTS = {
        "frame_interval_seconds": 0.1,
        "inspection_window_seconds": 10.0,
        "observation_temperature": 0.1,
        "fast_observation_mode": True,
        "stationary_motion_threshold": 1.8,
        "zone_crop_top_ratio": 0.2,
        "zone_crop_bottom_ratio": 0.85,
        "zone_presence_threshold": 12.0,
        "selector_min_stable_frames": 2,
        "selector_min_quality_score": 0.45,
        "selector_attempt_cooldown_seconds": 0.35,
        "selector_quality_improvement_delta": 0.08,
        "selector_max_attempts_per_board": 3,
        "selector_near_deadline_seconds": 0.7,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        instruction: str,
        vlm_client: UnifiedVLMClient,
        detection_agent: StreamlinedAgent,
        publisher_getter: Callable[[], Any],
        event_callback: Optional[Callable[[DetectionEvent, Dict[str, Any]], None]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.validate_instruction_scope(instruction)

        self.vlm_client = vlm_client
        self.detection_agent = detection_agent
        self._publisher_getter = publisher_getter
        self._event_callback = event_callback
        self._config = {**self.DEFAULTS, **(config or {})}
        self.context = MonitoringContext(instruction=instruction.strip())

        self.frame_interval = float(self._config.get("frame_interval_seconds", 0.25))
        self.observation_temperature = float(self._config.get("observation_temperature", 0.1))
        self.inspection_window_seconds = float(self._config.get("inspection_window_seconds", 10.0))
        self.fast_observation_mode = bool(
            self._config.get("fast_observation_mode", True)
        )
        self.stationary_motion_threshold = float(
            self._config.get("stationary_motion_threshold", 1.8)
        )
        self.zone_crop_top_ratio = float(self._config.get("zone_crop_top_ratio", 0.2))
        self.zone_crop_bottom_ratio = float(self._config.get("zone_crop_bottom_ratio", 0.85))
        self.zone_presence_threshold = float(self._config.get("zone_presence_threshold", 12.0))
        self.selector_min_stable_frames = int(self._config.get("selector_min_stable_frames", 2))
        self.selector_min_quality_score = float(
            self._config.get("selector_min_quality_score", 0.45)
        )
        self.selector_attempt_cooldown_seconds = float(
            self._config.get("selector_attempt_cooldown_seconds", 0.35)
        )
        self.selector_quality_improvement_delta = float(
            self._config.get("selector_quality_improvement_delta", 0.08)
        )
        self.selector_max_attempts_per_board = int(
            self._config.get("selector_max_attempts_per_board", 3)
        )
        self.selector_near_deadline_seconds = float(
            self._config.get("selector_near_deadline_seconds", 0.7)
        )

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._publisher = None
        self._subscriber_id = f"proactive_{int(time.time()*1000)}"
        self._last_frame_ts = 0.0
        self._last_motion_estimate = True
        self._prev_gray_roi: Optional[np.ndarray] = None
        self._prev_inspection_roi: Optional[np.ndarray] = None
        self._stable_frame_streak = 0
        self._inspection_attempts = 0
        self._last_inspection_attempt_ts = 0.0
        self._last_attempt_quality = 0.0
        self._best_frame: Optional[np.ndarray] = None
        self._best_frame_score = 0.0
        self._best_frame_metrics: Dict[str, float] = {}
        self._last_fsm_state = ConveyorState.MOVING.value
        self._cv_observation_failures = 0
        self._fsm = ConveyorInspectionFSM(
            inspection_window_seconds=self.inspection_window_seconds,
            on_decision=self._on_fsm_decision,
        )

    def start(self) -> None:
        """Start the proactive monitoring agent."""
        if self._running:
            logger.info("Proactive monitoring already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="ProactiveMonitoringAgent",
        )
        self._thread.start()
        logger.info("Proactive monitoring agent started (%s)", self._subscriber_id)

    def stop(self) -> None:
        """Stop the proactive monitoring agent."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._publisher:
            try:
                self._publisher.unsubscribe(self._subscriber_id)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        logger.info("Proactive monitoring agent stopped")

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self._running

    def update_instruction(self, instruction: str) -> None:
        """Update the monitoring instruction."""
        self.validate_instruction_scope(instruction)
        self.context.instruction = instruction.strip()
        logger.info("Updated proactive instruction: %s", self.context.instruction)

    def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the agent state."""
        return {
            "config": {
                "frame_interval_seconds": self.frame_interval,
                "inspection_window_seconds": self.inspection_window_seconds,
                "observation_temperature": self.observation_temperature,
                "fast_observation_mode": self.fast_observation_mode,
                "stationary_motion_threshold": self.stationary_motion_threshold,
                "selector_min_stable_frames": self.selector_min_stable_frames,
                "selector_min_quality_score": self.selector_min_quality_score,
                "selector_attempt_cooldown_seconds": self.selector_attempt_cooldown_seconds,
                "selector_quality_improvement_delta": (
                    self.selector_quality_improvement_delta
                ),
                "selector_max_attempts_per_board": (
                    self.selector_max_attempts_per_board
                ),
            },
            "context": self.context.snapshot(),
            "fsm": self._fsm.snapshot(),
            "running": self._running,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the monitoring agent."""
        total_decisions = self.context.inspections_completed
        return {
            "frames_processed": self.context.frames_processed,
            "inspections_started": self.context.inspections_started,
            "inspections_completed": self.context.inspections_completed,
            "defect_found": self.context.defect_found_count,
            "no_defect": self.context.no_defect_count,
            "unique_boards_inspected": len(self._fsm.inspected_boards),
            "decision_rate": (
                total_decisions / self.context.frames_processed
                if self.context.frames_processed > 0
                else 0.0
            ),
            "fsm_state": self._fsm.snapshot()["state"],
            "seconds_since_last_action": (
                time.time() - self.context.last_action_time
                if self.context.last_action_time
                else 0.0
            ),
        }

    def _loop(self) -> None:  # pylint: disable=too-many-branches,too-many-locals
        self._publisher = self._publisher_getter()
        if not self._publisher.subscribe(self._subscriber_id):
            logger.error("Proactive agent failed to subscribe to camera feed")
            self._running = False
            return

        try:
            while self._running:
                frame_obj = self._publisher.get_frame(
                    self._subscriber_id, timeout=0.25
                )
                if not frame_obj:
                    continue

                now = time.time()
                fsm_state = self._fsm.snapshot()["state"]
                if fsm_state != ConveyorState.INSPECTING.value:
                    if now - self._last_frame_ts < self.frame_interval:
                        continue
                self._last_frame_ts = now

                observation = self._observe(frame_obj)
                if not observation:
                    continue

                self.context.frames_processed += 1
                self.context.last_observation = observation

                self._fsm.update_inputs(
                    motion_moving=observation.motion_moving,
                    board_in_zone=observation.board_in_zone,
                    board_signature=observation.board_signature,
                )

                if (
                    self._fsm.begin_inspection_on_first_frame()
                ):
                    self._reset_frame_selector()
                    self.context.inspections_started += 1
                    logger.info(
                        "FSM transition: %s -> %s for board=%s",
                        ConveyorState.STOPPED_PENDING_INSPECTION.value,
                        ConveyorState.INSPECTING.value,
                        self._fsm.snapshot().get("active_board_signature"),
                    )

                current_state = self._fsm.snapshot()["state"]
                is_inspecting = (
                    current_state == ConveyorState.INSPECTING.value
                )
                was_inspecting = (
                    self._last_fsm_state == ConveyorState.INSPECTING.value
                )
                if is_inspecting and not was_inspecting:
                    self._reset_frame_selector()

                if is_inspecting:
                    selected_frame, selection_meta = self._select_inspection_frame(frame_obj)
                    if selected_frame is None:
                        self._last_fsm_state = current_state
                        continue
                    inspection_event = self._run_inspection_analysis(
                        selected_frame, selection_meta
                    )
                    self._fsm.submit_analysis(inspection_event)
                self._last_fsm_state = current_state
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Proactive monitoring loop crashed: %s", exc, exc_info=True)
        finally:
            try:
                self._publisher.unsubscribe(self._subscriber_id)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def _observe(self, frame_obj) -> Optional[ObservationResult]:
        """Observe the frame and extract relevant information."""
        return self._observe_fast_cv(frame_obj)

    def _observe_fast_cv(  # pylint: disable=too-many-locals
        self, frame_obj
    ) -> Optional[ObservationResult]:
        """Fast CV-only observation path for low-latency conveyor gating."""
        try:
            frame = frame_obj.raw_frame
            if frame is None:
                return None

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            h, w = gray.shape[:2]
            top = int(max(0, min(h - 1, h * self.zone_crop_top_ratio)))
            bottom = int(max(top + 1, min(h, h * self.zone_crop_bottom_ratio)))
            roi = gray[top:bottom, :w]
            if roi.size == 0:
                return None

            roi_small = cv2.resize(roi, (160, 120), interpolation=cv2.INTER_AREA)
            roi_blur = cv2.GaussianBlur(roi_small, (5, 5), 0)

            motion_score = 255.0
            motion_moving = self._last_motion_estimate
            if self._prev_gray_roi is not None and self._prev_gray_roi.shape == roi_blur.shape:
                diff = cv2.absdiff(roi_blur, self._prev_gray_roi)
                motion_score = float(np.mean(diff))
                motion_moving = motion_score > self.stationary_motion_threshold

            edge_map = cv2.Canny(roi_blur, 50, 150)
            edge_density = float(np.mean(edge_map > 0) * 100.0)
            board_in_zone = edge_density >= self.zone_presence_threshold

            board_signature = None
            if board_in_zone:
                active_signature = self._fsm.snapshot().get("active_board_signature")
                if active_signature and not motion_moving:
                    board_signature = str(active_signature)
                else:
                    board_signature = self._derive_board_signature(frame)

            self._prev_gray_roi = roi_blur
            self._last_motion_estimate = motion_moving

            self._cv_observation_failures = 0
            state_label = "moving" if motion_moving else "stopped"
            summary = (
                f"cv_fast_observation state={state_label} "
                f"motion_score={motion_score:.2f} edge_density={edge_density:.2f}"
            )
            return ObservationResult(
                frame_number=frame_obj.frame_number,
                timestamp=frame_obj.timestamp,
                scene_summary=summary,
                target_present=board_in_zone,
                target_state=state_label,
                confidence=0.9,
                motion_moving=motion_moving,
                board_in_zone=board_in_zone,
                board_signature=board_signature,
                raw_response="{\"source\":\"cv_fast\"}",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._cv_observation_failures += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Fast CV observation failed (%d): %s",
                    self._cv_observation_failures, exc
                )
            return None

    def _select_inspection_frame(  # pylint: disable=too-many-locals
        self, frame_obj
    ) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
        frame = frame_obj.raw_frame
        metrics = self._compute_frame_metrics(frame)
        is_stable = metrics["motion_score"] <= self.stationary_motion_threshold

        if is_stable:
            self._stable_frame_streak += 1
        else:
            self._stable_frame_streak = 0

        quality_score = metrics["quality_score"]
        if quality_score >= self._best_frame_score:
            self._best_frame_score = quality_score
            self._best_frame = frame.copy()
            self._best_frame_metrics = metrics

        now = time.time()
        can_attempt = (
            (now - self._last_inspection_attempt_ts)
            >= self.selector_attempt_cooldown_seconds
        )
        if self._inspection_attempts >= self.selector_max_attempts_per_board or not can_attempt:
            return None, {
                "attempted": False,
                "reason": "attempt_budget_or_cooldown",
                "metrics": metrics,
                "stable_frame_streak": self._stable_frame_streak,
                "best_frame_score": self._best_frame_score,
            }

        near_deadline = False
        remaining = self._fsm.snapshot().get("inspection_window_remaining_seconds")
        if isinstance(remaining, (float, int)):
            near_deadline = float(remaining) <= self.selector_near_deadline_seconds

        quality_ready = quality_score >= self.selector_min_quality_score
        stability_ready = (
            self._stable_frame_streak >= self.selector_min_stable_frames
        )
        improved_enough = quality_score >= (
            self._last_attempt_quality + self.selector_quality_improvement_delta
        )

        should_attempt = False
        if self._inspection_attempts == 0 and stability_ready and quality_ready:
            should_attempt = True
        elif stability_ready and quality_ready and improved_enough:
            should_attempt = True
        elif near_deadline and self._best_frame is not None:
            should_attempt = True

        if not should_attempt:
            return None, {
                "attempted": False,
                "reason": "quality_or_stability_not_ready",
                "metrics": metrics,
                "stable_frame_streak": self._stable_frame_streak,
                "best_frame_score": self._best_frame_score,
                "near_deadline": near_deadline,
            }

        selected_frame = frame
        selected_metrics = metrics
        use_best = (
            near_deadline and self._best_frame is not None
            and self._best_frame_score > quality_score
        )
        if use_best:
            selected_frame = self._best_frame
            selected_metrics = self._best_frame_metrics or metrics

        self._inspection_attempts += 1
        self._last_inspection_attempt_ts = now
        self._last_attempt_quality = float(
            selected_metrics.get("quality_score", quality_score)
        )
        return selected_frame, {
            "attempted": True,
            "reason": "selected_by_deterministic_quality_gate",
            "metrics": selected_metrics,
            "stable_frame_streak": self._stable_frame_streak,
            "attempt_number": self._inspection_attempts,
            "near_deadline": near_deadline,
            "remaining_seconds": remaining,
        }

    def _compute_frame_metrics(  # pylint: disable=too-many-locals
        self, frame: np.ndarray
    ) -> Dict[str, float]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]
        top = int(max(0, min(h - 1, h * self.zone_crop_top_ratio)))
        bottom = int(max(top + 1, min(h, h * self.zone_crop_bottom_ratio)))
        roi = gray[top:bottom, :w]
        roi_small = cv2.resize(roi, (160, 120), interpolation=cv2.INTER_AREA)
        roi_blur = cv2.GaussianBlur(roi_small, (5, 5), 0)

        lap = cv2.Laplacian(roi_blur, cv2.CV_64F)
        sharpness = float(np.var(lap))
        brightness = float(np.mean(roi_blur))
        contrast = float(np.std(roi_blur))

        motion_score = 255.0
        if (
            self._prev_inspection_roi is not None
            and self._prev_inspection_roi.shape == roi_blur.shape
        ):
            motion_score = float(
                np.mean(cv2.absdiff(roi_blur, self._prev_inspection_roi))
            )
        self._prev_inspection_roi = roi_blur

        exposure_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        sharpness_score = min(1.0, sharpness / 120.0)
        contrast_score = min(1.0, contrast / 45.0)
        motion_stability_score = max(
            0.0,
            1.0 - (
                motion_score / max(1.0, self.stationary_motion_threshold * 2.0)
            )
        )

        quality_score = (
            0.40 * sharpness_score
            + 0.25 * exposure_score
            + 0.20 * contrast_score
            + 0.15 * motion_stability_score
        )

        return {
            "quality_score": float(quality_score),
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "motion_score": motion_score,
            "sharpness_score": sharpness_score,
            "exposure_score": exposure_score,
            "contrast_score": contrast_score,
            "motion_stability_score": motion_stability_score,
        }

    def _reset_frame_selector(self) -> None:
        """Reset the frame selector state."""
        self._prev_inspection_roi = None
        self._stable_frame_streak = 0
        self._inspection_attempts = 0
        self._last_inspection_attempt_ts = 0.0
        self._last_attempt_quality = 0.0
        self._best_frame = None
        self._best_frame_score = 0.0
        self._best_frame_metrics = {}

    def _run_inspection_analysis(
        self, frame: np.ndarray, selection_meta: Dict[str, Any]
    ) -> Optional[DetectionEvent]:
        """Run inspection analysis on the selected frame."""
        if self._fsm.snapshot()["state"] != ConveyorState.INSPECTING.value:
            return None

        event = self.detection_agent.analyze_with_prompt(
            frame=frame,
            task_type=TaskType.CUSTOM,
            custom_prompt=self._pcb_defect_prompt(),
        )
        if event:
            event.decision_trace.setdefault("proactive_context", {})
            event.decision_trace["proactive_context"].update({
                "fsm_state": self._fsm.snapshot()["state"],
                "active_board_signature": (
                    self._fsm.snapshot().get("active_board_signature")
                ),
                "inspection_window_seconds": self.inspection_window_seconds,
                "frame_selection": selection_meta,
            })
        return event

    def _on_fsm_decision(self, payload: ConveyorDecisionPayload) -> None:
        self.context.inspections_completed += 1
        self.context.last_action_time = time.time()
        self.context.last_decision = payload.decision.value
        self.context.last_decision_reason = payload.reason
        self.context.board_decisions[payload.board_signature] = payload.decision.value

        if payload.decision == ConveyorDecision.DEFECT_FOUND:
            self.context.defect_found_count += 1
            event = payload.event
            if not event:
                event = DetectionEvent(
                    timestamp=datetime.now().isoformat(),
                    confidence=0.5,
                    primary_label="defect_found",
                    full_response=(
                        "FSM emitted DEFECT_FOUND without downstream "
                        "event payload."
                    ),
                    vision_description="Defect detected during inspection.",
                    decision_trace={},
                    detected=True,
                    should_alert=True,
                )
        else:
            self.context.no_defect_count += 1
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                confidence=1.0,
                primary_label="no_defect",
                full_response=(
                    "Inspection window expired without defect detection."
                ),
                vision_description=(
                    "No defect detected within enforced 10-second "
                    "inspection window."
                ),
                decision_trace={},
                detected=False,
                should_alert=False,
            )

        event.decision_trace.setdefault("proactive_context", {})
        event.decision_trace["proactive_context"].update({
            "fsm_decision": payload.decision.value,
            "fsm_reason": payload.reason,
            "board_signature": payload.board_signature,
            "fsm_state": ConveyorState.INSPECTED_COMPLETE.value,
            "inspection_window_seconds": self.inspection_window_seconds,
        })

        if self._event_callback:
            metadata = {
                "reason": "proactive_conveyor_fsm",
                "event_type": payload.decision.value,
                "board_signature": payload.board_signature,
                "fsm_state": ConveyorState.INSPECTED_COMPLETE.value,
            }
            self._event_callback(event, metadata)

    def _derive_motion_state(self, target_state: str) -> bool:
        if target_state in {"moving", "entering"}:
            self._last_motion_estimate = True
            return True
        if target_state in {"stopped", "stable"}:
            self._last_motion_estimate = False
            return False
        return self._last_motion_estimate

    @staticmethod
    def _derive_board_signature(frame: np.ndarray) -> str:
        """Deterministic board signature derived from image hash (non-LLM)."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
            avg = float(resized.mean())
            bits = (resized > avg).astype(np.uint8).flatten()
            packed = np.packbits(bits)
            digest = hashlib.sha256(packed.tobytes()).hexdigest()[:24]
            return f"board-{digest}"
        except Exception:  # pylint: disable=broad-exception-caught
            fallback = hashlib.sha256(frame.tobytes()).hexdigest()[:24]
            return f"board-{fallback}"

    @staticmethod
    def _pcb_defect_prompt() -> str:
        """Return the PCB defect inspection prompt."""
        return (
            "Inspect this PCB for manufacturing defects such as missing or misaligned "
            "components, solder bridges, insufficient solder joints, lifted pads, damaged "
            "traces, and contamination. Return whether a defect is present."
        )

    @classmethod
    def validate_instruction_scope(cls, instruction: str) -> None:
        """Validate that the instruction is within scope for PCB inspection."""
        result = get_classifier().classify(instruction or "")
        if result.domain == "pcb":
            return
        raise ValueError(f"{cls.SCOPE_REFUSAL_MESSAGE} Received domain: {result.domain}.")
