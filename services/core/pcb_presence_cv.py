"""Deterministic OpenCV PCB presence detector.

Shared by proactive verification tools and camera UI overlays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """Configuration for OpenCV PCB presence detection."""

    stationary_motion_threshold: float = 1.8
    zone_crop_top_ratio: float = 0.2
    zone_crop_bottom_ratio: float = 0.85
    zone_presence_threshold: float = 12.0
    segmentation_min_area_ratio: float = 0.04
    segmentation_max_area_ratio: float = 0.92
    segmentation_min_extent: float = 0.35
    presence_confirm_frames: int = 2
    absence_confirm_frames: int = 3


@dataclass
class CvPresenceResult:
    """Single-frame PCB presence result."""

    pcb_present: bool
    pcb_present_raw: bool
    motion_moving: bool
    motion_score: float
    edge_density: float
    candidate_area_ratio: float
    candidate_extent: float
    candidate_score: float
    processing_ms: float


class OpenCvPcbPresenceDetector:
    """Deterministic OpenCV PCB presence detector used for gating."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._prev_gray_roi: Optional[np.ndarray] = None
        self._last_motion_estimate = True
        self._board_present_state = False
        self._presence_streak = 0
        self._absence_streak = 0

    def reset_state(self) -> None:
        """Reset temporal state used for motion and debounce."""
        self._prev_gray_roi = None
        self._last_motion_estimate = True
        self._board_present_state = False
        self._presence_streak = 0
        self._absence_streak = 0

    def process_frame(self, frame: np.ndarray) -> CvPresenceResult:
        """Run OpenCV PCB presence detection on one frame."""
        started = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        top = int(max(0, min(h - 1, h * self.config.zone_crop_top_ratio)))
        bottom = int(max(top + 1, min(h, h * self.config.zone_crop_bottom_ratio)))
        roi = gray[top:bottom, :w]
        if roi.size == 0:
            elapsed_ms = ((cv2.getTickCount() - started) / cv2.getTickFrequency()) * 1000.0
            return CvPresenceResult(
                pcb_present=False,
                pcb_present_raw=False,
                motion_moving=False,
                motion_score=0.0,
                edge_density=0.0,
                candidate_area_ratio=0.0,
                candidate_extent=0.0,
                candidate_score=0.0,
                processing_ms=elapsed_ms,
            )

        roi_small = cv2.resize(roi, (160, 120), interpolation=cv2.INTER_AREA)
        roi_blur = cv2.GaussianBlur(roi_small, (5, 5), 0)

        motion_score = 255.0
        motion_moving = self._last_motion_estimate
        if self._prev_gray_roi is not None and self._prev_gray_roi.shape == roi_blur.shape:
            diff = cv2.absdiff(roi_blur, self._prev_gray_roi)
            motion_score = float(np.mean(diff))
            motion_moving = motion_score > self.config.stationary_motion_threshold

        edge_map = cv2.Canny(roi_blur, 50, 150)
        edge_density = float(np.mean(edge_map > 0) * 100.0)

        candidate = self._segment_board_candidate(roi_blur)
        raw_present = bool(candidate) and (
            edge_density >= max(2.0, self.config.zone_presence_threshold * 0.25)
            or (
                candidate is not None
                and candidate.get("area_ratio", 0.0)
                >= (self.config.segmentation_min_area_ratio * 1.5)
            )
        )
        debounced_present = self._debounce_presence(raw_present)

        self._prev_gray_roi = roi_blur
        self._last_motion_estimate = motion_moving

        elapsed_ms = ((cv2.getTickCount() - started) / cv2.getTickFrequency()) * 1000.0
        return CvPresenceResult(
            pcb_present=debounced_present,
            pcb_present_raw=raw_present,
            motion_moving=motion_moving,
            motion_score=motion_score,
            edge_density=edge_density,
            candidate_area_ratio=float(candidate.get("area_ratio", 0.0)) if candidate else 0.0,
            candidate_extent=float(candidate.get("extent", 0.0)) if candidate else 0.0,
            candidate_score=float(candidate.get("score", 0.0)) if candidate else 0.0,
            processing_ms=elapsed_ms,
        )

    def _segment_board_candidate(
        self,
        roi_blur: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        edges = cv2.Canny(roi_blur, 35, 120)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        roi_area = float(roi_blur.shape[0] * roi_blur.shape[1])
        best: Optional[Dict[str, float]] = None
        best_score = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            area_ratio = area / max(1.0, roi_area)
            if area_ratio < self.config.segmentation_min_area_ratio:
                continue
            if area_ratio > self.config.segmentation_max_area_ratio:
                continue

            x, y, w_box, h_box = cv2.boundingRect(contour)
            rect_area = float(max(1, w_box * h_box))
            extent = area / rect_area
            if extent < self.config.segmentation_min_extent:
                continue

            aspect_ratio = float(w_box) / float(max(1, h_box))
            if aspect_ratio < 0.25 or aspect_ratio > 6.0:
                continue

            score = (0.65 * area_ratio) + (0.35 * min(1.0, extent))
            if score >= best_score:
                best_score = score
                best = {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w_box),
                    "h": float(h_box),
                    "area_ratio": float(area_ratio),
                    "extent": float(extent),
                    "score": float(score),
                }

        return best

    def _debounce_presence(self, present_raw: bool) -> bool:
        if present_raw:
            self._presence_streak += 1
            self._absence_streak = 0
            if self._presence_streak >= self.config.presence_confirm_frames:
                self._board_present_state = True
        else:
            self._absence_streak += 1
            self._presence_streak = 0
            if self._absence_streak >= self.config.absence_confirm_frames:
                self._board_present_state = False
        return self._board_present_state
