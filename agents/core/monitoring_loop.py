"""Proactive monitoring loop — observation infrastructure.

This is the ONLY deterministic component in the agent system. It:

1. Acquires frames from the camera publisher
2. Computes CV sensor readings (motion, edge density, board presence)
3. Detects meaningful state changes (board stopped in zone)
4. Fires ``on_board_ready`` so the **LLM decides** what to do

Everything else — inspection, alerting, classification, logging — is
decided by the LLM through MCP tool calls.

Architecture
~~~~~~~~~~~~
::

    Camera Feed
        │
        ▼
    MonitoringLoop._observe()       ← CV sensor readings (deterministic)
        │
        ▼
    State-change detected?
        │ YES
        ▼
    on_board_ready(frame, context)  ← LLM decides tools to call

The CV observation code (motion detection, board segmentation, presence
debouncing) produces **sensor readings**, not decisions.  They answer
"what is happening?" — never "what should we do?".
"""

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
from agents.core.state import DetectionEvent

logger = get_logger(__name__)


# ── Observation dataclass ─────────────────────────────────────────────────


@dataclass
class Observation:
    """CV sensor readings from a single frame — NO decisions."""

    frame_number: int
    timestamp: str
    motion_score: float
    motion_moving: bool
    edge_density: float
    board_in_zone: bool
    board_signature: Optional[str]
    frame_quality: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "motion_score": round(self.motion_score, 3),
            "motion_moving": self.motion_moving,
            "edge_density": round(self.edge_density, 3),
            "board_in_zone": self.board_in_zone,
            "board_signature": self.board_signature,
            "frame_quality": {
                k: round(v, 3) for k, v in self.frame_quality.items()
            },
        }


# ── Monitoring context ────────────────────────────────────────────────────


@dataclass
class MonitoringContext:
    """Runtime context for observability and metrics."""

    instruction: str
    frames_processed: int = 0
    inspections_completed: int = 0
    defect_found_count: int = 0
    no_defect_count: int = 0
    last_observation: Optional[Observation] = None
    last_action_time: float = 0.0
    board_decisions: Dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "frames_processed": self.frames_processed,
            "inspections_completed": self.inspections_completed,
            "defect_found_count": self.defect_found_count,
            "no_defect_count": self.no_defect_count,
            "last_observation": (
                self.last_observation.to_dict()
                if self.last_observation
                else None
            ),
            "last_action_time": self.last_action_time,
            "board_decisions": self.board_decisions,
        }


# ── Monitoring loop ──────────────────────────────────────────────────────


class MonitoringLoop:
    """Frame acquisition + CV observation + state-change detection.

    This is the ONLY deterministic component.  The loop:

    1. Acquires frames from the camera publisher
    2. Computes CV sensor readings (motion, edge density, board presence)
    3. Detects state changes (board stopped → inspection opportunity)
    4. Fires *on_board_ready* for the LLM to decide what to do

    Parameters
    ----------
    instruction : str
        The PCB inspection instruction (scope-validated on init).
    publisher_getter : callable
        Returns the camera frame publisher.
    on_board_ready : callable(frame, observation_context)
        Called when a board stops in the inspection zone.
        The LLM / MCP system decides what tools to invoke.
    event_callback : optional callable(DetectionEvent, metadata)
        For recording completed inspection results to DB / websocket.
    config : optional dict
        CV sensor tuning parameters (thresholds, not decisions).
    """

    SCOPE_REFUSAL_MESSAGE = (
        "Refused: proactive monitoring accepts only PCB manufacturing "
        "defect inspection instructions."
    )

    DEFAULTS: Dict[str, Any] = {
        "frame_interval_seconds": 0.1,
        "stationary_motion_threshold": 1.8,
        "zone_crop_top_ratio": 0.2,
        "zone_crop_bottom_ratio": 0.85,
        "zone_presence_threshold": 12.0,
        "segmentation_min_area_ratio": 0.04,
        "segmentation_max_area_ratio": 0.92,
        "segmentation_min_extent": 0.35,
        "presence_confirm_frames": 2,
        "absence_confirm_frames": 3,
        "track_iou_threshold": 0.22,
        "track_max_lost_frames": 5,
    }

    def __init__(
        self,
        *,
        instruction: str,
        publisher_getter: Callable[[], Any],
        on_board_ready: Callable[[np.ndarray, Dict[str, Any]], None],
        event_callback: Optional[
            Callable[[DetectionEvent, Dict[str, Any]], None]
        ] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.validate_instruction_scope(instruction)

        cfg: Dict[str, Any] = {**self.DEFAULTS, **(config or {})}
        self.context = MonitoringContext(instruction=instruction.strip())
        self._publisher_getter = publisher_getter
        self._on_board_ready = on_board_ready
        self._event_callback = event_callback

        # CV sensor parameters (tuning, not decisions)
        self._frame_interval = float(cfg["frame_interval_seconds"])
        self._motion_threshold = float(cfg["stationary_motion_threshold"])
        self._zone_top = float(cfg["zone_crop_top_ratio"])
        self._zone_bottom = float(cfg["zone_crop_bottom_ratio"])
        self._edge_threshold = float(cfg["zone_presence_threshold"])
        self._seg_min_area = float(cfg["segmentation_min_area_ratio"])
        self._seg_max_area = float(cfg["segmentation_max_area_ratio"])
        self._seg_min_extent = float(cfg["segmentation_min_extent"])
        self._presence_confirm = int(cfg["presence_confirm_frames"])
        self._absence_confirm = int(cfg["absence_confirm_frames"])
        self._iou_threshold = float(cfg["track_iou_threshold"])
        self._max_lost = int(cfg["track_max_lost_frames"])

        # Thread / loop state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._publisher: Optional[Any] = None
        self._subscriber_id = f"monitor_{int(time.time() * 1000)}"
        self._last_frame_ts = 0.0

        # CV observation state
        self._prev_gray_roi: Optional[np.ndarray] = None
        self._last_motion_moving = True
        self._board_present = False
        self._presence_streak = 0
        self._absence_streak = 0
        self._active_track_id: Optional[int] = None
        self._next_track_id = 1
        self._tracked_bbox: Optional[tuple[int, int, int, int]] = None
        self._track_lost_frames = 0

        # State-change tracking (replaces hardcoded FSM)
        self._prev_board_stopped = False
        self._inspected_signatures: set[str] = set()
        self._max_inspected_signatures = 100_000

        # Frame storage state (instance-level, not class-level)
        self._last_store_ts: float = 0.0
        self._last_stored_gray: Optional[np.ndarray] = None
        self._last_stored_time: float = 0.0

    # ── Public interface ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the monitoring loop in a background thread."""
        if self.is_running:
            logger.info("Monitoring loop already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="MonitoringLoop",
        )
        self._thread.start()
        logger.info("Monitoring loop started (%s)", self._subscriber_id)

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "Monitoring loop thread did not stop within timeout"
                )
        # Unsubscription is handled by the _loop() finally block.
        logger.info("Monitoring loop stopped")

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and (
            self._thread is not None and self._thread.is_alive()
        )

    def update_instruction(self, instruction: str) -> None:
        """Update the monitoring instruction (scope-validated)."""
        self.validate_instruction_scope(instruction)
        self.context.instruction = instruction.strip()
        logger.info("Updated instruction: %s", self.context.instruction)

    def snapshot(self) -> Dict[str, Any]:
        """Runtime snapshot for dashboards and the LLM context."""
        return {
            "running": self.is_running,
            "context": self.context.snapshot(),
            "inspected_board_count": len(self._inspected_signatures),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the monitoring loop."""
        return {
            "frames_processed": self.context.frames_processed,
            "inspections_completed": self.context.inspections_completed,
            "defect_found": self.context.defect_found_count,
            "no_defect": self.context.no_defect_count,
            "unique_boards_inspected": len(self._inspected_signatures),
        }

    # ── Main loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        self._publisher = self._publisher_getter()
        if not self._publisher or not self._publisher.subscribe(
            self._subscriber_id
        ):
            logger.error("Failed to subscribe to camera feed")
            self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                frame_obj = self._acquire_frame()
                if frame_obj is None:
                    continue

                obs = self._observe(frame_obj)
                if obs is None:
                    continue

                self.context.frames_processed += 1
                self.context.last_observation = obs

                # Store frames when PCB present + low motion (mechanical)
                if obs.board_in_zone and not obs.motion_moving:
                    self._store_frame(frame_obj, obs)

                # ── State-change detection (replaces hardcoded FSM) ───
                board_stopped = obs.board_in_zone and not obs.motion_moving
                just_stopped = board_stopped and not self._prev_board_stopped
                self._prev_board_stopped = board_stopped

                if just_stopped and obs.board_signature:
                    if obs.board_signature not in self._inspected_signatures:
                        self._inspected_signatures.add(obs.board_signature)
                        # Evict oldest entries if set grows too large
                        if len(self._inspected_signatures) > self._max_inspected_signatures:
                            # Remove ~10% of entries to amortise eviction cost
                            to_remove = self._max_inspected_signatures // 10
                            it = iter(self._inspected_signatures)
                            for _ in range(to_remove):
                                self._inspected_signatures.discard(next(it))
                        self._fire_board_ready(frame_obj.raw_frame, obs)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Monitoring loop crashed: %s", exc, exc_info=True
            )
        finally:
            if self._publisher:
                try:
                    self._publisher.unsubscribe(self._subscriber_id)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                self._publisher = None

    def _acquire_frame(self):
        """Fetch and pace frame retrieval."""
        if self._publisher is None:
            return None
        frame_obj = self._publisher.get_frame(
            self._subscriber_id, timeout=0.25
        )
        if not frame_obj:
            return None
        now = time.time()
        if (now - self._last_frame_ts) < self._frame_interval:
            return None
        self._last_frame_ts = now
        return frame_obj

    def _fire_board_ready(
        self, frame: np.ndarray, obs: Observation
    ) -> None:
        """Delegate the inspection decision to the LLM / callback."""
        context = {
            "trigger": "board_stopped",
            "observation": obs.to_dict(),
            "instruction": self.context.instruction,
            "boards_inspected": len(self._inspected_signatures),
        }
        try:
            self._on_board_ready(frame, context)
            self.context.last_action_time = time.time()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("on_board_ready callback failed: %s", exc)

    # ── CV observation (sensor readings, not decisions) ───────────────

    def _observe(self, frame_obj) -> Optional[Observation]:
        """Extract CV sensor readings from a single frame."""
        try:
            frame = frame_obj.raw_frame
            if frame is None:
                return None

            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )
            h, w = gray.shape[:2]
            top = int(max(0, min(h - 1, h * self._zone_top)))
            bottom = int(max(top + 1, min(h, h * self._zone_bottom)))
            roi = gray[top:bottom, :w]
            if roi.size == 0:
                return None

            roi_small = cv2.resize(
                roi, (160, 120), interpolation=cv2.INTER_AREA
            )
            roi_blur = cv2.GaussianBlur(roi_small, (5, 5), 0)

            # ── Motion sensor ─────────────────────────────────────────
            motion_score = 255.0
            motion_moving = self._last_motion_moving
            if (
                self._prev_gray_roi is not None
                and self._prev_gray_roi.shape == roi_blur.shape
            ):
                diff = cv2.absdiff(roi_blur, self._prev_gray_roi)
                motion_score = float(np.mean(diff))
                motion_moving = motion_score > self._motion_threshold
            self._prev_gray_roi = roi_blur
            self._last_motion_moving = motion_moving

            # ── Edge density sensor ───────────────────────────────────
            edge_map = cv2.Canny(roi_blur, 50, 150)
            edge_density = float(np.mean(edge_map > 0) * 100.0)

            # ── Board presence sensor ─────────────────────────────────
            candidate = self._segment_board(roi_blur)
            raw_present = bool(candidate) and (
                edge_density >= max(2.0, self._edge_threshold * 0.25)
                or (
                    candidate is not None
                    and candidate.get("area_ratio", 0)
                    >= self._seg_min_area * 1.5
                )
            )
            board_in_zone = self._debounce_presence(raw_present)

            # ── Board tracking ────────────────────────────────────────
            if board_in_zone and candidate:
                x, y, w_box, h_box = candidate["bbox"]
                sig = self._update_tracker(
                    (x, y + top, w_box, h_box), motion_moving
                )
            else:
                sig = self._update_tracker(None, motion_moving)

            if board_in_zone and not sig:
                sig = self._derive_signature(frame)

            # ── Frame quality metrics ─────────────────────────────────
            quality = self._compute_quality(roi_blur)

            return Observation(
                frame_number=frame_obj.frame_number,
                timestamp=frame_obj.timestamp,
                motion_score=motion_score,
                motion_moving=motion_moving,
                edge_density=edge_density,
                board_in_zone=board_in_zone,
                board_signature=sig,
                frame_quality=quality,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("CV observation failed: %s", exc)
            return None

    # ── CV helpers (pure computation) ─────────────────────────────────

    def _segment_board(
        self, roi_blur: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Find the most board-like contour in the ROI."""
        edges = cv2.Canny(roi_blur, 35, 120)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        roi_area = float(roi_blur.shape[0] * roi_blur.shape[1])
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue
            ratio = area / max(1.0, roi_area)
            if not (self._seg_min_area <= ratio <= self._seg_max_area):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            extent = area / float(max(1, bw * bh))
            if extent < self._seg_min_extent:
                continue
            aspect = float(bw) / float(max(1, bh))
            if aspect < 0.25 or aspect > 6.0:
                continue
            score = 0.65 * ratio + 0.35 * min(1.0, extent)
            if score >= best_score:
                best_score = score
                best = {
                    "area_ratio": float(ratio),
                    "extent": float(extent),
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                }
        return best

    def _debounce_presence(self, raw: bool) -> bool:
        """Hysteresis on raw board-present signal."""
        if raw:
            self._presence_streak = min(
                self._presence_streak + 1, self._presence_confirm
            )
            self._absence_streak = 0
            if self._presence_streak >= self._presence_confirm:
                self._board_present = True
        else:
            self._absence_streak = min(
                self._absence_streak + 1, self._absence_confirm
            )
            self._presence_streak = 0
            if self._absence_streak >= self._absence_confirm:
                self._board_present = False
        return self._board_present

    def _update_tracker(
        self,
        bbox: Optional[tuple[int, int, int, int]],
        motion_moving: bool,
    ) -> Optional[str]:
        """Track a board across frames and return its signature."""
        if bbox is None:
            self._track_lost_frames += 1
            if self._track_lost_frames > self._max_lost:
                self._active_track_id = None
                self._tracked_bbox = None
            if self._active_track_id is None:
                return None
            return f"track-{self._active_track_id}"

        if self._active_track_id is None:
            self._active_track_id = self._next_track_id
            self._next_track_id += 1
            self._tracked_bbox = bbox
            self._track_lost_frames = 0
            return f"track-{self._active_track_id}"

        iou = self._iou(self._tracked_bbox, bbox)
        if iou < self._iou_threshold and motion_moving:
            self._active_track_id = self._next_track_id
            self._next_track_id += 1
        self._tracked_bbox = bbox
        self._track_lost_frames = 0
        return f"track-{self._active_track_id}"

    @staticmethod
    def _iou(
        box_a: Optional[tuple[int, int, int, int]],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """IoU between two (x, y, w, h) bounding boxes."""
        if box_a is None:
            return 0.0
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = float(aw * ah + bw * bh) - inter
        return inter / max(1.0, union)

    @staticmethod
    def _derive_signature(frame: np.ndarray) -> str:
        """Perceptual hash fallback for board identification."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            norm = cv2.equalizeHist(gray)
            resized = cv2.resize(
                norm, (32, 32), interpolation=cv2.INTER_AREA
            )
            dct = cv2.dct(resized.astype(np.float32))
            avg = float(np.mean(dct[:8, :8][1:, 1:]))
            bits = (dct[:8, :8] > avg).astype(np.uint8).flatten()
            digest = hashlib.sha256(
                np.packbits(bits).tobytes()
            ).hexdigest()[:24]
            return f"board-{digest}"
        except Exception:  # pylint: disable=broad-exception-caught
            fallback = hashlib.sha256(frame.tobytes()).hexdigest()[:24]
            return f"board-{fallback}"

    @staticmethod
    def _compute_quality(roi_blur: np.ndarray) -> Dict[str, float]:
        """Frame quality metrics — reported to the LLM, not used for gating."""
        lap = cv2.Laplacian(roi_blur, cv2.CV_64F)
        return {
            "sharpness": float(np.var(lap)),
            "brightness": float(np.mean(roi_blur)),
            "contrast": float(np.std(roi_blur)),
        }

    # ── Frame storage (mechanical, not a decision) ────────────────────

    _STORE_COOLDOWN = 2.0
    _DEDUP_THRESHOLD = 0.92
    _DEDUP_WINDOW = 15.0

    def _store_frame(self, frame_obj, obs: Observation) -> None:
        """Persist frame to disk + database for LLM tool access."""
        now = time.time()
        if (now - self._last_store_ts) < self._STORE_COOLDOWN:
            return
        frame = frame_obj.raw_frame
        if frame is None:
            return

        # Similarity-based dedup
        thumb = self._thumbnail(frame)
        if thumb is not None and self._last_stored_gray is not None:
            if (now - self._last_stored_time) <= self._DEDUP_WINDOW:
                sim = self._similarity(self._last_stored_gray, thumb)
                if sim >= self._DEDUP_THRESHOLD:
                    return

        self._last_store_ts = now
        try:
            import os  # pylint: disable=import-outside-toplevel
            from pathlib import Path  # pylint: disable=import-outside-toplevel

            store_dir = (
                Path(os.getenv("CAMERA_AGENT_DATA_DIR", "."))
                / "pcb_frame_store"
            )
            store_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = store_dir / f"pcb_{ts}.jpg"
            cv2.imwrite(str(path), frame)

            from app.database import PCBFrameStoreRepository  # pylint: disable=import-outside-toplevel

            PCBFrameStoreRepository.store(
                image_path=str(path),
                motion_score=obs.motion_score,
                board_signature=obs.board_signature or "",
                frame_number=obs.frame_number,
                quality_score=obs.frame_quality.get("sharpness", 0.0),
            )
            self._last_stored_gray = thumb
            self._last_stored_time = now
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to store frame: %s", exc)

    @staticmethod
    def _thumbnail(frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            return cv2.cvtColor(
                cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        try:
            return float(
                cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0][0]
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return 0.0

    # ── Scope validation ──────────────────────────────────────────────

    @classmethod
    def validate_instruction_scope(cls, instruction: str) -> None:
        """Reject non-PCB-inspection instructions.

        Uses the LLM classifier to determine if the instruction is
        within the PCB inspection scope.
        """
        text = (instruction or "").strip()
        if not text:
            raise ValueError(cls.SCOPE_REFUSAL_MESSAGE)

        result = get_classifier().classify(text)
        if result.domain == "pcb":
            return
        raise ValueError(
            f"{cls.SCOPE_REFUSAL_MESSAGE} Received domain: {result.domain}."
        )
