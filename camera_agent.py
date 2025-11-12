#!/usr/bin/env python3
"""
ZEDEDA Camera Monitoring Agent - Packaging Box Detection
Monitors /dev/video0 and sends alerts when an unlabeled packaging box is detected.
"""

import base64
import copy
import json
import logging
import math
import mimetypes
import os
import smtplib
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import cv2
import torch
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
DEFAULT_LOG_FILE = os.getenv("CAMERA_AGENT_LOG_FILE", "camera_agent.log")
log_handlers = [logging.StreamHandler(sys.stdout)]

if DEFAULT_LOG_FILE:
    log_path = Path(DEFAULT_LOG_FILE).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_path, mode='a'))
    except OSError as exc:  # pragma: no cover - best effort warning path
        sys.stderr.write(
            f"Warning: unable to use log file {log_path}: {exc}\n"
        )

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.getenv("CAMERA_AGENT_CONFIG", "config.yaml")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert assorted truthy/falsey representations to bool with a default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


@dataclass
class DetectionEvent:
    """Represents a packaging box detection event."""
    timestamp: str
    confidence: float
    primary_label: str
    full_response: str
    image_path: Optional[str] = None
    vision_description: str = ""
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False
    shipping_label_present: Optional[bool] = None
    should_alert: bool = True
    tools_used: List[str] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ShippingLabelAnalysis:
    """Structured result from the shipping label analyzer."""
    detected: bool
    presence_confidence: float
    candidate_count: int
    average_aspect_ratio: float
    summary: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    region_count: int = 0
    cluster_confidences: List[float] = field(default_factory=list)
    cluster_method: str = "kmeans"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "presence_confidence": round(float(self.presence_confidence), 3),
            "candidate_count": int(self.candidate_count),
            "average_aspect_ratio": round(float(self.average_aspect_ratio), 3),
            "summary": self.summary,
            "candidates": self.candidates,
            "region_count": int(self.region_count),
            "cluster_confidences": [round(float(conf), 3) for conf in self.cluster_confidences],
            "cluster_method": self.cluster_method,
        }


@dataclass
class PackagingBoxAnalysis:
    """Structured result from the classical packaging box analyzer."""
    detected: bool
    confidence: float
    candidate_count: int
    average_aspect_ratio: float
    average_rectangularity: float
    average_solidity: float
    summary: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    estimated_box_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": round(float(self.confidence), 3),
            "candidate_count": int(self.candidate_count),
            "average_aspect_ratio": round(float(self.average_aspect_ratio), 3),
            "average_rectangularity": round(float(self.average_rectangularity), 3),
            "average_solidity": round(float(self.average_solidity), 3),
            "summary": self.summary,
            "candidates": self.candidates,
            "estimated_box_count": int(self.estimated_box_count),
        }


class ShippingLabelAnalyzer:
    """Lightweight CV + clustering model to identify shipping label regions."""

    def __init__(self) -> None:
        self.min_area_ratio = 0.001
        self.max_area_ratio = 0.25
        self.min_fill_ratio = 0.5
        self.aspect_range = (0.45, 6.0)
        self.color_min_area_ratio = 0.0006
        self.yellow_hue_range = (15, 45)
        self.white_value_threshold = 200
        self.white_saturation_max = 40
        self.min_color_region_area = 180
        self.max_cluster_candidates = 6

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return minimum

    def _cluster_candidates(
        self,
        candidates: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Cluster candidate regions using k-means to estimate label groups."""
        if not candidates:
            return {
                "region_count": 0,
                "cluster_confidences": [],
                "labels": None,
                "method": "kmeans"
            }

        height, width = image_shape
        max_dim = float(max(width, height, 1))
        features: List[List[float]] = []

        for cand in candidates:
            bbox = cand.get("bounding_box", [0, 0, 0, 0])
            x, y, w_box, h_box = bbox if len(bbox) == 4 else (0, 0, 0, 0)
            cx = x + w_box / 2.0
            cy = y + h_box / 2.0
            area_ratio = cand.get("area_ratio_raw", cand.get("area_ratio", 0.0))
            norm_cx = cx / max_dim
            norm_cy = cy / max_dim
            norm_area = float(area_ratio) * 8.0  # emphasize spatial clusters with size weighting
            features.append([norm_cx, norm_cy, norm_area])

        data = np.array(features, dtype=np.float32)
        if data.shape[0] == 1:
            return {
                "region_count": 1,
                "cluster_confidences": [candidates[0].get("score", 0.5)],
                "labels": np.array([0], dtype=np.int32),
                "method": "kmeans"
            }

        max_k = min(self.max_cluster_candidates, data.shape[0])
        best_k = 1
        best_penalty = float("inf")
        best_result = {
            "labels": np.zeros((data.shape[0], 1), dtype=np.int32),
            "center": None,
            "compactness": 0.0
        }

        for k in range(1, max_k + 1):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
            try:
                compactness, labels, centers = cv2.kmeans(
                    data,
                    k,
                    None,
                    criteria,
                    5,
                    cv2.KMEANS_PP_CENTERS
                )
            except Exception:
                continue

            penalty = compactness + (k ** 1.5) * 0.08
            if penalty < best_penalty:
                best_penalty = penalty
                best_k = k
                best_result = {
                    "labels": labels,
                    "centers": centers,
                    "compactness": compactness
                }

        labels = best_result.get("labels")
        if labels is None:
            labels = np.zeros((data.shape[0], 1), dtype=np.int32)

        cluster_confidences: List[float] = []
        for cluster_id in range(best_k):
            cluster_members = [
                candidates[idx]
                for idx, label in enumerate(labels.flatten())
                if label == cluster_id
            ]
            if not cluster_members:
                cluster_confidences.append(0.0)
                continue
            top_score = max(member.get("score", 0.0) for member in cluster_members)
            mean_score = sum(member.get("score", 0.0) for member in cluster_members) / len(cluster_members)
            cluster_confidences.append(self._clamp(0.65 * top_score + 0.35 * mean_score))

        cluster_confidences.sort(reverse=True)
        return {
            "region_count": best_k,
            "cluster_confidences": cluster_confidences,
            "labels": labels,
            "method": "kmeans"
        }

    def _score_candidate(
        self,
        *,
        area_ratio: float,
        aspect_ratio: float,
        fill_ratio: float,
        mean_intensity: float,
        mean_hsv: Optional[List[float]],
        source: str = "contour"
    ) -> Dict[str, float]:
        eps = 1e-6
        aspect_score = 0.0
        if aspect_ratio > eps:
            aspect_score = math.exp(-abs(math.log(aspect_ratio)))
            aspect_score = self._clamp(aspect_score)

        fill_score = self._clamp(fill_ratio)
        area_score = self._clamp((area_ratio - self.min_area_ratio) / max(self.max_area_ratio - self.min_area_ratio, eps))
        brightness_score = self._clamp(mean_intensity / 200.0)

        hue = sat = val = 0.0
        if mean_hsv and len(mean_hsv) == 3:
            hue, sat, val = mean_hsv

        yellow_score = 0.0
        if self.yellow_hue_range[0] <= hue <= self.yellow_hue_range[1] and sat >= 60 and val >= 110:
            yellow_score = self._clamp((sat - 60) / 140.0) * 0.6 + self._clamp((val - 110) / 145.0) * 0.4

        white_score = 0.0
        if val >= self.white_value_threshold and sat <= self.white_saturation_max + 20:
            white_score = self._clamp((val - self.white_value_threshold) / 55.0)

        color_score = max(yellow_score, white_score)

        combined_color = max(color_score, brightness_score)
        base_score = (
            0.3 * area_score
            + 0.25 * aspect_score
            + 0.3 * fill_score
            + 0.15 * combined_color
        )

        if source == "color_mask":
            base_score = self._clamp(base_score + 0.08)

        return {
            "area": round(area_score, 3),
            "aspect": round(aspect_score, 3),
            "fill": round(fill_score, 3),
            "color": round(combined_color, 3),
            "base": round(self._clamp(base_score), 3)
        }

    def analyze(
        self,
        frame: Optional[np.ndarray],
        gray: Optional[np.ndarray],
        hsv: Optional[np.ndarray]
    ) -> Optional[ShippingLabelAnalysis]:
        try:
            if frame is None or gray is None or hsv is None:
                return None

            if gray.ndim != 2 or hsv.ndim != 3:
                return None

            height, width = gray.shape[:2]
            if height == 0 or width == 0:
                return None

            image_area = float(width * height)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

            edges = cv2.Canny(normalized, 50, 150, apertureSize=3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates: List[Dict[str, Any]] = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue
                area_ratio = area / image_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue

                rect = cv2.minAreaRect(contour)
                (w, h) = rect[1]
                if w == 0 or h == 0:
                    continue
                width_px = max(w, h)
                height_px = min(w, h)
                aspect_ratio = width_px / height_px if height_px else 0.0
                if not (self.aspect_range[0] <= aspect_ratio <= self.aspect_range[1]):
                    continue

                x, y, w_box, h_box = cv2.boundingRect(contour)
                if w_box == 0 or h_box == 0:
                    continue
                fill_ratio = area / float(w_box * h_box)
                if fill_ratio < self.min_fill_ratio:
                    continue

                region = gray[y:y + h_box, x:x + w_box]
                mean_intensity = float(region.mean()) if region.size else 0.0
                region_hsv = hsv[y:y + h_box, x:x + w_box]
                mean_hsv = [float(region_hsv[:, :, idx].mean()) if region_hsv.size else 0.0 for idx in range(3)]
                score_components = self._score_candidate(
                    area_ratio=area_ratio,
                    aspect_ratio=aspect_ratio,
                    fill_ratio=fill_ratio,
                    mean_intensity=mean_intensity,
                    mean_hsv=mean_hsv,
                    source="contour"
                )

                candidates.append({
                    "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                    "aspect_ratio": round(aspect_ratio, 3),
                    "fill_ratio": round(fill_ratio, 3),
                    "mean_intensity": round(mean_intensity, 1),
                    "area_ratio": round(area_ratio, 4),
                    "area_ratio_raw": float(area_ratio),
                    "mean_hsv": [round(channel, 1) for channel in mean_hsv],
                    "source": "contour",
                    "score": score_components["base"],
                    "score_components": score_components
                })

            candidate_count = len(candidates)
            if not candidate_count:
                color_mask = np.zeros_like(gray, dtype=np.uint8)

                lower_yellow = np.array([self.yellow_hue_range[0], 70, 80])
                upper_yellow = np.array([self.yellow_hue_range[1], 255, 255])
                yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                lower_white = np.array([0, 0, self.white_value_threshold])
                upper_white = np.array([180, self.white_saturation_max, 255])
                white_mask = cv2.inRange(hsv, lower_white, upper_white)

                color_mask = cv2.bitwise_or(yellow_mask, white_mask)
                if np.count_nonzero(color_mask) > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
                    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in color_contours:
                        area = cv2.contourArea(contour)
                        if area < self.min_color_region_area:
                            continue
                        area_ratio = area / image_area
                        if area_ratio < self.color_min_area_ratio:
                            continue

                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        if w_box == 0 or h_box == 0:
                            continue

                        long_side = max(w_box, h_box)
                        short_side = min(w_box, h_box)
                        aspect_ratio = long_side / short_side if short_side else 0.0
                        if aspect_ratio < self.aspect_range[0] or aspect_ratio > self.aspect_range[1]:
                            continue

                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
                        solidity = area / hull_area if hull_area else 0.0
                        if solidity < self.min_fill_ratio:
                            continue

                        color_region = hsv[y:y + h_box, x:x + w_box]
                        mean_hue = float(color_region[:, :, 0].mean()) if color_region.size else 0.0
                        mean_sat = float(color_region[:, :, 1].mean()) if color_region.size else 0.0
                        mean_val = float(color_region[:, :, 2].mean()) if color_region.size else 0.0

                        region_gray = gray[y:y + h_box, x:x + w_box]
                        mean_intensity = float(region_gray.mean()) if region_gray.size else 0.0
                        score_components = self._score_candidate(
                            area_ratio=area_ratio,
                            aspect_ratio=aspect_ratio,
                            fill_ratio=solidity,
                            mean_intensity=mean_intensity,
                            mean_hsv=[mean_hue, mean_sat, mean_val],
                            source="color_mask"
                        )

                        candidates.append({
                            "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                            "aspect_ratio": round(aspect_ratio, 3),
                            "fill_ratio": round(solidity, 3),
                            "mean_hsv": [round(mean_hue, 1), round(mean_sat, 1), round(mean_val, 1)],
                            "mean_intensity": round(mean_intensity, 1),
                            "area_ratio": round(area_ratio, 4),
                            "area_ratio_raw": float(area_ratio),
                            "source": "color_mask",
                            "score": score_components["base"],
                            "score_components": score_components
                        })

            candidate_count = len(candidates)
            if candidate_count:
                avg_aspect = sum(c.get("aspect_ratio", 0.0) for c in candidates) / candidate_count
                cluster_summary = self._cluster_candidates(candidates, (height, width))
                region_count = cluster_summary.get("region_count", 0)
                cluster_confidences = cluster_summary.get("cluster_confidences", [])
                max_cluster_conf = cluster_confidences[0] if cluster_confidences else 0.0
                avg_top_clusters = sum(cluster_confidences[:3]) / max(1, min(3, len(cluster_confidences))) if cluster_confidences else 0.0
                density_score = self._clamp(region_count / 4.0)
                color_hits = sum(1 for c in candidates if c.get("source") == "color_mask")
                color_density = self._clamp(color_hits / candidate_count) if candidate_count else 0.0
                presence_confidence = self._clamp(
                    0.5 * max_cluster_conf +
                    0.25 * avg_top_clusters +
                    0.15 * density_score +
                    0.1 * color_density
                )
            else:
                avg_aspect = 0.0
                presence_confidence = 0.02
                cluster_summary = {"region_count": 0, "cluster_confidences": [], "method": "kmeans"}
                max_cluster_conf = 0.0
                region_count = 0

            detected = candidate_count > 0 and max_cluster_conf >= 0.45
            summary = (
                f"{candidate_count} candidate label region{'s' if candidate_count != 1 else ''} clustered into "
                f"{cluster_summary.get('region_count', 0)} group{'s' if cluster_summary.get('region_count', 0) != 1 else ''} "
                f"(confidence {presence_confidence:.2f})"
            )

            labels = cluster_summary.get("labels")
            if isinstance(labels, np.ndarray) and labels.size == candidate_count:
                label_values = labels.flatten().tolist()
                for idx, cluster_id in enumerate(label_values):
                    candidates[idx]["cluster_id"] = int(cluster_id)

            return ShippingLabelAnalysis(
                detected=detected,
                presence_confidence=presence_confidence,
                candidate_count=candidate_count,
                average_aspect_ratio=avg_aspect,
                summary=summary,
                candidates=candidates,
                region_count=cluster_summary.get("region_count", 0),
                cluster_confidences=cluster_summary.get("cluster_confidences", []),
                cluster_method=cluster_summary.get("method", "kmeans")
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Shipping label analyzer failed: %s", exc, exc_info=True)
            return None


class PackagingBoxAnalyzer:
    """Classical vision pipeline to flag packaging-box shaped regions."""

    def __init__(self) -> None:
        self.min_area_ratio = 0.004
        self.max_area_ratio = 0.7
        self.aspect_range = (0.3, 4.2)
        self.min_rectangularity = 0.5
        self.min_solidity = 0.55
        self.min_vertices = 4
        self.max_vertices = 10
        self.min_color_area_ratio = 0.002
        self.min_edge_density = 0.02

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return minimum

    def _score_candidate(
        self,
        *,
        area_ratio: float,
        aspect_ratio: float,
        rectangularity: float,
        solidity: float,
        edge_density: float,
        mean_hsv: Optional[List[float]],
        source: str = "contour"
    ) -> Dict[str, float]:
        eps = 1e-6
        area_score = self._clamp((area_ratio - self.min_area_ratio) / max(self.max_area_ratio - self.min_area_ratio, eps))

        target_aspect = 1.6
        aspect_score = 0.0
        if aspect_ratio > eps:
            aspect_score = math.exp(-abs(math.log(aspect_ratio / target_aspect)))
            aspect_score = self._clamp(aspect_score)

        rectangularity_score = self._clamp(rectangularity)
        solidity_score = self._clamp(solidity)
        edge_score = self._clamp(edge_density * 2.0)

        hue = sat = val = 0.0
        if mean_hsv and len(mean_hsv) == 3:
            hue, sat, val = mean_hsv

        brown_score = 0.0
        if 5 <= hue <= 40 and sat >= 45 and 50 <= val <= 225:
            hue_center = 22.5
            hue_span = 17.5
            hue_component = self._clamp(1.0 - abs(hue - hue_center) / hue_span)
            sat_component = self._clamp((sat - 45) / 140.0)
            val_component = self._clamp((val - 50) / 175.0)
            brown_score = (0.5 * hue_component + 0.3 * sat_component + 0.2 * val_component)

        color_score = max(brown_score, edge_score)

        base_score = (
            0.25 * area_score +
            0.25 * rectangularity_score +
            0.2 * solidity_score +
            0.15 * aspect_score +
            0.15 * color_score
        )

        if source == "color_mask":
            base_score = self._clamp(base_score + 0.07)

        return {
            "area": round(area_score, 3),
            "rectangularity": round(rectangularity_score, 3),
            "solidity": round(solidity_score, 3),
            "aspect": round(aspect_score, 3),
            "color": round(color_score, 3),
            "base": round(self._clamp(base_score), 3)
        }

    def analyze(
        self,
        frame: Optional[np.ndarray],
        gray: Optional[np.ndarray],
        hsv: Optional[np.ndarray]
    ) -> Optional[PackagingBoxAnalysis]:
        try:
            if frame is None or gray is None or hsv is None:
                return None

            if gray.ndim != 2 or hsv.ndim != 3:
                return None

            height, width = gray.shape[:2]
            if height == 0 or width == 0:
                return None

            image_area = float(width * height)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 35, 120)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            dilated = cv2.dilate(closed, kernel, iterations=1)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: List[Dict[str, Any]] = []

            for contour in contours:
                area = cv2.contourArea(contour)
                if area <= 0:
                    continue

                area_ratio = area / image_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(contour)
                if w_box == 0 or h_box == 0:
                    continue

                long_side = max(w_box, h_box)
                short_side = min(w_box, h_box)
                aspect_ratio = long_side / short_side if short_side else 0.0
                if not (self.aspect_range[0] <= aspect_ratio <= self.aspect_range[1]):
                    continue

                rectangularity = area / float(w_box * h_box)
                if rectangularity < self.min_rectangularity:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull) if len(hull) >= 3 else 0.0
                solidity = area / hull_area if hull_area else 0.0
                if solidity < self.min_solidity:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
                vertex_count = len(approx)
                if vertex_count < self.min_vertices or vertex_count > self.max_vertices:
                    continue

                region_edges = edges[y:y + h_box, x:x + w_box]
                edge_density = float(region_edges.mean()) / 255.0 if region_edges.size else 0.0
                if edge_density < self.min_edge_density:
                    continue

                region_hsv = hsv[y:y + h_box, x:x + w_box]
                mean_hsv = [float(region_hsv[:, :, idx].mean()) if region_hsv.size else 0.0 for idx in range(3)]
                score_components = self._score_candidate(
                    area_ratio=area_ratio,
                    aspect_ratio=aspect_ratio,
                    rectangularity=rectangularity,
                    solidity=solidity,
                    edge_density=edge_density,
                    mean_hsv=mean_hsv,
                    source="contour"
                )

                candidates.append({
                    "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                    "area_ratio": round(area_ratio, 4),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "rectangularity": round(rectangularity, 3),
                    "solidity": round(solidity, 3),
                    "edge_density": round(edge_density, 3),
                    "mean_hsv": [round(channel, 1) for channel in mean_hsv],
                    "score": score_components["base"],
                    "score_components": score_components,
                    "vertex_count": int(vertex_count),
                    "source": "contour"
                })

            if not candidates:
                lower_brown = np.array([5, 60, 40])
                upper_brown = np.array([25, 180, 200])
                brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                lower_cardboard = np.array([10, 50, 60])
                upper_cardboard = np.array([30, 180, 230])
                cardboard_mask = cv2.inRange(hsv, lower_cardboard, upper_cardboard)
                color_mask = cv2.bitwise_or(brown_mask, cardboard_mask)
                if np.count_nonzero(color_mask) > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
                    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in color_contours:
                        area = cv2.contourArea(contour)
                        if area <= 0:
                            continue
                        area_ratio = area / image_area
                        if area_ratio < self.min_color_area_ratio:
                            continue
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        if w_box == 0 or h_box == 0:
                            continue
                        long_side = max(w_box, h_box)
                        short_side = min(w_box, h_box)
                        aspect_ratio = long_side / short_side if short_side else 0.0
                        if not (self.aspect_range[0] <= aspect_ratio <= self.aspect_range[1]):
                            continue
                        rectangularity = area / float(w_box * h_box)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
                        solidity = area / hull_area if hull_area else 0.0
                        region_edges = edges[y:y + h_box, x:x + w_box]
                        edge_density = float(region_edges.mean()) / 255.0 if region_edges.size else 0.0
                        region_hsv = hsv[y:y + h_box, x:x + w_box]
                        mean_hsv = [float(region_hsv[:, :, idx].mean()) if region_hsv.size else 0.0 for idx in range(3)]
                        score_components = self._score_candidate(
                            area_ratio=area_ratio,
                            aspect_ratio=aspect_ratio,
                            rectangularity=rectangularity,
                            solidity=solidity,
                            edge_density=edge_density,
                            mean_hsv=mean_hsv,
                            source="color_mask"
                        )
                        candidates.append({
                            "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                            "area_ratio": round(area_ratio, 4),
                            "aspect_ratio": round(aspect_ratio, 3),
                            "rectangularity": round(rectangularity, 3),
                            "solidity": round(solidity, 3),
                            "edge_density": round(edge_density, 3),
                            "mean_hsv": [round(channel, 1) for channel in mean_hsv],
                            "score": score_components["base"],
                            "score_components": score_components,
                            "vertex_count": len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)),
                            "source": "color_mask"
                        })

            candidate_count = len(candidates)
            if candidate_count:
                avg_aspect = sum(c["aspect_ratio"] for c in candidates) / candidate_count
                avg_rectangularity = sum(c["rectangularity"] for c in candidates) / candidate_count
                avg_solidity = sum(c["solidity"] for c in candidates) / candidate_count
                top_score = max(c["score"] for c in candidates)
                top_scores = sorted((c["score"] for c in candidates), reverse=True)[:3]
                avg_top = sum(top_scores) / len(top_scores)
                density = self._clamp(candidate_count / 2.5)
                color_density = self._clamp(
                    sum(1 for c in candidates if c.get("source") == "color_mask") / candidate_count
                )
                confidence = self._clamp(
                    0.5 * top_score +
                    0.25 * avg_top +
                    0.15 * density +
                    0.1 * color_density
                )

                strong_candidates = [c for c in candidates if c.get("score", 0.0) >= 0.58]
                moderate_candidates = [c for c in candidates if c.get("score", 0.0) >= 0.45]
                if strong_candidates:
                    estimated_box_count = len(strong_candidates)
                elif moderate_candidates:
                    estimated_box_count = len(moderate_candidates)
                else:
                    estimated_box_count = 1 if top_score >= 0.45 else 0
            else:
                avg_aspect = 0.0
                avg_rectangularity = 0.0
                avg_solidity = 0.0
                confidence = 0.1
                top_score = 0.0
                estimated_box_count = 0

            detected = candidate_count > 0 and confidence >= 0.45
            summary = (
                f"{candidate_count} packaging-like region{'s' if candidate_count != 1 else ''} "
                f"(confidence {confidence:.2f}, top score {top_score:.2f}, "
                f"estimated boxes {estimated_box_count})"
            )

            return PackagingBoxAnalysis(
                detected=detected,
                confidence=confidence,
                candidate_count=candidate_count,
                average_aspect_ratio=avg_aspect,
                average_rectangularity=avg_rectangularity,
                average_solidity=avg_solidity,
                summary=summary,
                candidates=candidates,
                estimated_box_count=estimated_box_count
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Packaging analyzer failed: %s", exc, exc_info=True)
            return None


class RFDetrPackageDetector:
    """Wrapper around the RF-DETR Medium package detector from Hugging Face."""

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold
        self.repo_id = "Mact0/rf-detr-package-detection"
        self.checkpoint_filename = "checkpoint_best_total.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            try:
                from rfdetr import RFDETRMedium  # local import to avoid mandatory dependency at module import

                checkpoint_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.checkpoint_filename,
                    repo_type="model"
                )

                model = RFDETRMedium()
                model.model.reinitialize_detection_head(num_classes=2)
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=False
                )
                weights = state_dict.get("model", state_dict)

                base_module = model.model.model
                missing = base_module.load_state_dict(weights, strict=False)
                if isinstance(missing, tuple) and any(missing):
                    logger.debug("RF-DETR load: missing=%s unexpected=%s", missing[0], missing[1])

                base_module.to(self.device)
                base_module.eval()
                model.model.device = self.device
                model.device = self.device
                self._model = model
                logger.info("RF-DETR package detector initialized on %s", self.device)
            except FileNotFoundError as exc:
                logger.error("RF-DETR checkpoint not found: %s", exc)
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.error("Failed to initialize RF-DETR package detector: %s", exc, exc_info=True)

    def analyze(self, frame: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return None

        if self._model is None:
            self._load_model()

        if self._model is None:
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            detections = self._model.predict(image, threshold=self.threshold)

            if detections is None:
                return None

            if isinstance(detections, list):
                detections = detections[0] if detections else None
                if detections is None:
                    return None

            boxes = detections.xyxy if hasattr(detections, "xyxy") else None
            confidences = detections.confidence if hasattr(detections, "confidence") else None
            labels = detections.class_id if hasattr(detections, "class_id") else None

            count = int(boxes.shape[0]) if boxes is not None else 0
            if count == 0:
                return {
                    "box_count": 0,
                    "average_confidence": 0.0,
                    "max_confidence": 0.0,
                    "detections": [],
                    "summary": "RF-DETR detected no packaging boxes"
                }

            boxes_list: List[Dict[str, Any]] = []
            scores: List[float] = []
            for idx in range(count):
                x1, y1, x2, y2 = [float(val) for val in boxes[idx]]
                score = float(confidences[idx]) if confidences is not None else 0.0
                cls_id = int(labels[idx]) if labels is not None else 0
                boxes_list.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": round(score, 4),
                    "class_id": cls_id
                })
                scores.append(score)

            avg_conf = sum(scores) / len(scores) if scores else 0.0
            max_conf = max(scores) if scores else 0.0
            summary = (
                f"RF-DETR detected {count} package{'s' if count != 1 else ''} "
                f"(avg conf {avg_conf:.2f}, max {max_conf:.2f})"
            )

            return {
                "box_count": count,
                "average_confidence": round(avg_conf, 3),
                "max_confidence": round(max_conf, 3),
                "detections": boxes_list,
                "summary": summary
            }
        except Exception as exc:  # pragma: no cover - best effort logging path
            logger.error("RF-DETR inference failed: %s", exc, exc_info=True)
            return None

class OllamaVisionClient:
    """Client for interacting with Ollama vision models."""
    
    def __init__(self, base_url: str, model: str, timeout: int = 60):
        if not model:
            raise ValueError("Vision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        logger.info("Initialized Ollama client for vision model '%s' at %s", model, self.base_url)
    
    def analyze_image(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        """Send image to Ollama for analysis."""
        try:
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            
            logger.debug(f"Sending request to {self.base_url}/api/generate")
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Ollama response: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to analyze image with Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image analysis: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Ollama service."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=10)
            response.raise_for_status()
            logger.info("Ollama connection test successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False


class DecisionLLM:
    """Second-stage LLM for packaging box decisions with tool calling capability."""
    
    def __init__(self, base_url: str, model: str, timeout: int = 30):
        if not model:
            raise ValueError("Decision model name must be provided")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        user_agent = os.getenv("CAMERA_AGENT_USER_AGENT", "camera-agent/1.0")
        self.session.headers.update({"User-Agent": user_agent})
        self.tools = self._define_tools()
        logger.info("Initialized Decision LLM model '%s' at %s", model, self.base_url)
    
    def _define_tools(self) -> List[Dict]:
        """Define available tools for the decision LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "trigger_packaging_alert",
                    "description": (
                        "Use when at least one packaging/shipping box is present AND no shipping label is clearly visible. "
                        "Call this to trigger an alert about unlabeled packaging."  # noqa: E501
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "Confidence (0.0-1.0) that an unlabeled packaging box is present"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Concise justification for raising the alert"
                            },
                            "box_count": {
                                "type": "integer",
                                "description": "Estimated number of packaging boxes in view"
                            },
                            "label_count": {
                                "type": "integer",
                                "description": "Estimated number of shipping labels associated with those boxes"
                            },
                            "shipping_label_present": {
                                "type": "boolean",
                                "description": "Should normally be false; set true only if a label is visible"
                            },
                            "labels_per_box": {
                                "type": "number",
                                "description": "Average number of labels per detected box"
                            },
                            "box_description": {
                                "type": "string",
                                "description": "Optional location or appearance notes about the box"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional free-form observations"
                            }
                        },
                        "required": ["confidence", "reasoning", "box_count", "label_count"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "record_no_detection",
                    "description": (
                        "Use when no alert is needed: either no packaging boxes are present, or boxes have clear shipping labels."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of why no action is needed"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Optional confidence (0.0-1.0) that no boxes require action"
                            },
                            "shipping_label_present": {
                                "type": ["boolean", "null"],
                                "description": "Set true if labels are visible on any boxes, false if none are present"
                            },
                            "box_count": {
                                "type": "integer",
                                "description": "Estimated number of boxes (0 if none)"
                            },
                            "label_count": {
                                "type": "integer",
                                "description": "Estimated number of labels associated with the boxes"
                            },
                            "labels_per_box": {
                                "type": "number",
                                "description": "Average labels per detected box"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional supplemental notes"
                            }
                        },
                        "required": ["reasoning"]
                    }
                }
            }
        ]
    
    def make_detection_decision(self, vision_description: str, extra_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Request a unified tool call from the decision LLM and interpret the result."""
        tool_trace_entries: List[Dict[str, Any]] = []
        context_lines: List[str] = []
        shipping_hint = None
        packaging_hint = None

        if isinstance(extra_context, dict):
            shipping_hint = extra_context.get("shipping_label_hint")
            packaging_hint = extra_context.get("packaging_hint")

            if shipping_hint:
                context_lines.append(f"Local shipping label analysis: {shipping_hint}")
            if packaging_hint:
                context_lines.append(f"Classical packaging analysis: {packaging_hint}")

            box_count_ctx = extra_context.get("packaging_box_count")
            label_count_ctx = extra_context.get("label_region_count")
            labels_per_box_ctx = extra_context.get("labels_per_box")
            multi_label_flag_ctx = extra_context.get("multi_label_flag")

            metrics_parts: List[str] = []
            if box_count_ctx is not None:
                metrics_parts.append(f"boxes≈{box_count_ctx}")
            if label_count_ctx is not None:
                metrics_parts.append(f"labels≈{label_count_ctx}")
            if labels_per_box_ctx is not None:
                metrics_parts.append(f"labels_per_box≈{labels_per_box_ctx}")
            if metrics_parts:
                context_lines.append("Scene metrics: " + ", ".join(str(part) for part in metrics_parts))
            if multi_label_flag_ctx:
                context_lines.append("Multiple labels detected on individual boxes")

        system_prompt = (
            "You are the safety decision-maker for packaging detection. "
            "You MUST call exactly one tool to report the outcome. Choose based on the description:\n"
            "- trigger_packaging_alert: at least one packaging box AND no visible shipping label (alert required).\n"
            "- record_no_detection: no boxes, or boxes are present but labels are visible or the scene is inconclusive (no alert).\n"
            "Always include your reasoning and any box/label counts you can infer. Do not return plain text."
        )

        user_prompt = (
            "Vision AI description:\n"
            f"{vision_description}"
        )

        if context_lines:
            context_block = "\n".join(context_lines)
            user_prompt += f"\n\nAdditional context:\n{context_block}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "tools": self.tools,
            "stream": False
        }

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                converted = int(round(float(value)))
                return converted
            except (TypeError, ValueError):
                return None

        def _coerce_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y"}:
                    return True
                if normalized in {"false", "0", "no", "n"}:
                    return False
            return None

        try:
            logger.info("🤖 Decision LLM requesting unified tool call...")
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            assistant_message = result.get("message", {})
            tool_calls = assistant_message.get("tool_calls") or []

            logger.debug(
                "Decision LLM tool-call raw response: %s",
                json.dumps(result, indent=2) if isinstance(result, dict) else result
            )

            if not tool_calls:
                logger.warning("Decision LLM returned no tool calls; treating as no detection")
                trace_payload = {
                    "vision_description": vision_description,
                    "tool_error": "no_tool_call",
                    "tools_used": [],
                    "tool_trace": []
                }
                if context_lines:
                    trace_payload["context"] = context_lines
                return {
                    "detected": False,
                    "decision_trace": trace_payload
                }

            if len(tool_calls) > 1:
                logger.debug("Decision LLM returned multiple tool calls; only first will be used")

            tool_call = tool_calls[0]
            function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            tool_name = function_payload.get("name") or "unknown_tool"
            raw_arguments = function_payload.get("arguments", {})

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    logger.warning("Decision LLM returned non-JSON arguments: %s", raw_arguments)
                    arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}

            tool_trace_entry = {
                "name": tool_name,
                "arguments": arguments
            }
            tool_trace_entries.append(tool_trace_entry)
            tools_used = [tool_name]

            decision_trace: Dict[str, Any] = {
                "vision_description": vision_description,
                "tool_trace": tool_trace_entries,
                "tools_used": tools_used,
                "decision_action": tool_name
            }

            if context_lines:
                decision_trace["context"] = context_lines

            if shipping_hint is not None:
                decision_trace.setdefault("shipping_label_hint", shipping_hint)
            if packaging_hint is not None:
                decision_trace.setdefault("packaging_hint", packaging_hint)

            box_count_value = _coerce_int(arguments.get("box_count"))
            label_count_value = _coerce_int(arguments.get("label_count"))
            labels_per_box_value = _coerce_float(arguments.get("labels_per_box"), float('nan'))
            if math.isnan(labels_per_box_value):
                labels_per_box_value = None

            scene_metrics: Dict[str, Any] = {}
            if box_count_value is not None:
                scene_metrics["box_count"] = box_count_value
            if label_count_value is not None:
                scene_metrics["label_count"] = label_count_value
            if labels_per_box_value is not None:
                scene_metrics["labels_per_box"] = round(labels_per_box_value, 3)
            if arguments.get("notes"):
                scene_metrics["notes"] = arguments.get("notes")
            if scene_metrics:
                decision_trace["scene_metrics"] = scene_metrics

            def _finalize_common_trace(classification: str) -> None:
                decision_trace["classification"] = classification
                decision_trace["tool_arguments"] = arguments

            if tool_name == "trigger_packaging_alert":
                confidence = max(0.0, min(1.0, _coerce_float(arguments.get("confidence"), 0.85)))
                reasoning = arguments.get("reasoning") or "Unlabeled packaging box detected by decision model"
                shipping_label_present = _coerce_bool(arguments.get("shipping_label_present"))
                if shipping_label_present is None:
                    shipping_label_present = False

                _finalize_common_trace("BOX_NO_LABEL")
                decision_trace["reasoning"] = reasoning
                decision_trace["confidence"] = confidence
                decision_trace["shipping_label_present"] = shipping_label_present
                decision_trace["llm_reported_box_count"] = box_count_value
                decision_trace["llm_reported_label_count"] = label_count_value
                if arguments.get("box_description"):
                    decision_trace["box_description"] = arguments.get("box_description")

                logger.info("🚨 Decision LLM raised unlabeled packaging alert (confidence %.2f)", confidence)
                return {
                    "detected": True,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "shipping_label_present": shipping_label_present,
                    "should_alert": True,
                    "decision_trace": decision_trace,
                    "tools_used": tools_used,
                    "tool_trace": tool_trace_entries
                }

            if tool_name == "record_no_detection":
                confidence_val = arguments.get("confidence")
                confidence = _coerce_float(confidence_val, 0.5) if confidence_val is not None else 0.5
                reasoning = arguments.get("reasoning") or "No packaging boxes require action"

                shipping_label_present = _coerce_bool(arguments.get("shipping_label_present"))

                _finalize_common_trace("NO_BOX_DETECTED")
                decision_trace["reasoning"] = reasoning
                decision_trace["confidence"] = confidence
                if shipping_label_present is not None:
                    decision_trace["shipping_label_present"] = shipping_label_present
                decision_trace["llm_reported_box_count"] = box_count_value
                decision_trace["llm_reported_label_count"] = label_count_value

                logger.info("✅ Decision LLM recorded no detection (confidence %.2f)", confidence)

                has_boxes = box_count_value is not None and box_count_value > 0

                if has_boxes and shipping_label_present is True:
                    decision_trace["classification"] = "BOX_WITH_LABEL"
                    logger.info("📦 Decision LLM reports boxes with visible labels; no alert needed")
                    return {
                        "detected": True,
                        "reasoning": reasoning,
                        "confidence": confidence,
                        "shipping_label_present": True,
                        "should_alert": False,
                        "decision_trace": decision_trace,
                        "tools_used": tools_used,
                        "tool_trace": tool_trace_entries
                    }

                return {
                    "detected": False,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "shipping_label_present": shipping_label_present,
                    "should_alert": False,
                    "decision_trace": decision_trace,
                    "tools_used": tools_used,
                    "tool_trace": tool_trace_entries
                }

            logger.warning("Decision LLM returned unknown tool '%s'; treating as no detection", tool_name)
            decision_trace["tool_error"] = "unknown_tool"
            return {
                "detected": False,
                "decision_trace": decision_trace,
                "tools_used": tools_used,
                "tool_trace": tool_trace_entries
            }

        except Exception as exc:  # pragma: no cover - resilience path
            logger.error("Decision LLM error: %s", exc)
            trace_payload = {
                "error": str(exc),
                "vision_description": vision_description,
                "tools_used": [entry.get("name") for entry in tool_trace_entries] if tool_trace_entries else [],
                "tool_trace": tool_trace_entries
            }
            if context_lines:
                trace_payload["context"] = context_lines
            return {
                "detected": False,
                "decision_trace": trace_payload
            }
    
    def test_connection(self) -> bool:
        """Test connection to decision LLM."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with OK."}
                ],
                "stream": False
            }

            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            result = response.json()
            assistant_message = result.get("message", {})
            content = assistant_message.get("content", "")
            return isinstance(content, str) and "OK" in content.upper()

        except Exception as e:
            logger.error(f"Decision LLM connection test failed: {e}")
            return False


class AlertManager:
    """Handles alert notifications via email and other channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_config = config.get("notifications", {}).get("email", {})
        self.desktop_config = config.get("notifications", {}).get("desktop", {})
        logger.info("Alert manager initialized")

    def refresh_config(self, config: Dict[str, Any]) -> None:
        """Reload configuration references after an update."""
        self.config = config
        self.email_config = config.get("notifications", {}).get("email", {})
        self.desktop_config = config.get("notifications", {}).get("desktop", {})
    
    def send_email_alert(self, event: DetectionEvent, rule: Dict[str, Any]) -> bool:
        """Send email alert for detection event."""
        if not _coerce_bool(self.email_config.get("enabled"), False):
            logger.info("Email alerts disabled")
            return False
        
        try:
            # Create email message
            msg = EmailMessage()
            
            # Resolve email action configuration (supports legacy and new schema)
            actions_email = rule.get("actions", {}).get("email")
            legacy_email = rule.get("email", {})
            email_action = actions_email if actions_email is not None else legacy_email

            if not email_action:
                logger.warning(f"Rule {rule.get('id', 'unknown')} missing email configuration; skipping alert")
                return False

            if not _coerce_bool(email_action.get("enabled"), _coerce_bool(legacy_email.get("enabled"), True)):
                logger.info(f"Email action disabled for rule {rule.get('id', 'unknown')}")
                return False
            
            # Template substitution
            subject_template = (
                email_action.get("subject")
                or legacy_email.get("subject")
                or "Unlabeled Packaging Box Detected"
            )
            body_template = (
                email_action.get("body")
                or legacy_email.get("body")
                or "An unlabeled packaging box was detected in the camera feed."
            )
            
            # Replace template variables
            template_vars = {
                "timestamp": event.timestamp,
                "confidence": f"{event.confidence:.2f}",
                "primary_label": event.primary_label,
                "full_response": event.full_response,
                "device": f"/dev/video{os.getenv('CAMERA_INDEX', '0')}",
                "shipping_label_present": (
                    "yes" if event.shipping_label_present else "no" if event.shipping_label_present is not None else "unknown"
                ),
                "should_alert": str(event.should_alert).lower()
            }
            
            subject = subject_template
            body = body_template
            for key, value in template_vars.items():
                subject = subject.replace(f"{{{{{key}}}}}", str(value))
                body = body.replace(f"{{{{{key}}}}}", str(value))
            
            msg["Subject"] = subject
            
            # Use environment variables for email credentials
            sender_email = os.getenv("EMAIL_USER") or self.email_config.get("sender_email", "noreply@zededa.com")
            env_password = os.getenv("EMAIL_PASS")
            legacy_password = self.email_config.get("sender_password")

            if legacy_password and not env_password:
                logger.warning(
                    "Using sender_password from configuration file; migrate credentials to EMAIL_PASS."
                )

            sender_password = env_password or legacy_password or ""

            msg["From"] = os.getenv("EMAIL_FROM", sender_email)
            
            # Get recipient emails
            recipients = email_action.get("to") or legacy_email.get("to") or self.email_config.get("recipients") or []
            if isinstance(recipients, str):
                recipients = [recipients]
            if not recipients:
                logger.error("No email recipients configured in rule")
                return False
            
            msg["To"] = ", ".join(recipients)
            msg.set_content(body)

            if event.image_path:
                try:
                    image_path = Path(event.image_path).expanduser()
                    if image_path.exists():
                        mime_type, _ = mimetypes.guess_type(str(image_path))
                        if mime_type:
                            maintype, subtype = mime_type.split("/", 1)
                        else:
                            maintype, subtype = "application", "octet-stream"
                        with image_path.open("rb") as img_file:
                            img_bytes = img_file.read()
                        msg.add_attachment(
                            img_bytes,
                            maintype=maintype,
                            subtype=subtype,
                            filename=image_path.name
                        )
                        logger.debug("Attached detection image %s to email alert", image_path)
                    else:
                        logger.warning("Detection image path %s not found; skipping attachment", image_path)
                except Exception as exc:
                    logger.error("Failed to attach detection image %s: %s", event.image_path, exc)
            
            # Send email
            smtp_server = os.getenv("EMAIL_SMTP_SERVER") or self.email_config.get("smtp_server", "smtp.gmail.com")
            try:
                smtp_port = int(os.getenv("EMAIL_SMTP_PORT", self.email_config.get("smtp_port", 587)))
            except (TypeError, ValueError):
                logger.error("Invalid SMTP port configuration")
                return False

            use_tls_config = _coerce_bool(
                email_action.get("use_tls"),
                _coerce_bool(self.email_config.get("use_tls"), True)
            )
            use_tls = _coerce_bool(os.getenv("EMAIL_USE_TLS"), use_tls_config)
            
            if not sender_email or not sender_password:
                logger.error("Email credentials missing; set EMAIL_USER and EMAIL_PASS environment variables")
                return False
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent successfully to {msg['To']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_desktop_notification(self, event: DetectionEvent) -> bool:
        """Send desktop notification (if available)."""
        if not _coerce_bool(self.desktop_config.get("enabled"), False):
            logger.debug("Desktop notifications disabled")
            return False
        try:
            import plyer
            try:
                timeout = int(self.desktop_config.get("timeout", 10))
            except (TypeError, ValueError):
                timeout = 10
            title = self.desktop_config.get("title", "ZEDEDA Camera Alert")
            message_template = self.desktop_config.get(
                "message",
                "Unlabeled packaging box detected at {timestamp}"
            )
            label_status = (
                "yes" if event.shipping_label_present else "no" if event.shipping_label_present is not None else "unknown"
            )
            message = message_template.format(
                timestamp=event.timestamp,
                confidence=f"{event.confidence:.2f}",
                label=event.primary_label,
                shipping_label_present=label_status
            )
            plyer.notification.notify(
                title=title,
                message=message,
                timeout=timeout
            )
            logger.info("Desktop notification sent")
            return True
        except ImportError:
            logger.debug("plyer not available for desktop notifications")
            return False
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
            return False


class MonitorDetectionAgent:
    """Main agent class that orchestrates camera monitoring and alert processing."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
        self.config = self._load_config()
        self.last_error: Optional[str] = None
        
        # Initialize components
        ollama_cfg = self.config.get("ollama")
        if not isinstance(ollama_cfg, dict):
            raise ValueError("Configuration must include an 'ollama' section with model settings.")

        required_keys = ["url", "vision_model", "decision_model"]
        missing_keys = [key for key in required_keys if not ollama_cfg.get(key)]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Ollama configuration missing required keys: {missing}")

        ollama_url = str(ollama_cfg["url"]).rstrip("/")
        vision_model = str(ollama_cfg["vision_model"])
        decision_model = str(ollama_cfg["decision_model"])

        ollama_timeout = ollama_cfg.get("timeout", 60)
        try:
            ollama_timeout = int(ollama_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ollama timeout must be an integer value") from exc

        self.ollama_client = OllamaVisionClient(ollama_url, vision_model, timeout=ollama_timeout)
        self.decision_llm = DecisionLLM(ollama_url, decision_model)
        self.alert_manager = AlertManager(self.config)
        self.packaging_box_analyzer = PackagingBoxAnalyzer()
        self._rfdet_detector: Optional[RFDetrPackageDetector] = None
        self._rfdet_detector_failed = False
        self.images_dir = Path("detected_images")
        self.processed_frames_dir = Path("processed_frames")
        self.save_images = False
        self.save_processed_frames = False
        self.agent_ssim_threshold = 0.995
        self.agent_ssim_recheck_seconds = 4.0
        self.agent_ssim_cache_ttl_seconds = 30.0
        self._ssim_reference_size = (320, 240)
        self._last_similarity_frame = None
        self._last_processed_event: Optional[DetectionEvent] = None
        self._last_llm_run_time = 0.0
        self._last_cache_reset = time.time()
        self._apply_runtime_config()
        self.clear_similarity_cache("startup")

        logger.info("Packaging Detection Agent initialized for web integration")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config = self.load_config_from_path(self.config_path)
        logger.info(f"Configuration loaded from {self.config_path}")
        return config

    @classmethod
    def load_config_from_path(cls, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load and validate configuration data from disk."""
        target_path = Path(path or DEFAULT_CONFIG_PATH).expanduser()
        if not target_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {target_path}")

        with target_path.open('r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)

        if not isinstance(config, dict):
            raise ValueError(f"Configuration file must contain a mapping at the root: {target_path}")

        return config

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return the persisted configuration as the canonical default."""
        return copy.deepcopy(cls.load_config_from_path())

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _safe_clamped_float(cls, value: Any, default: float, minimum: float, maximum: float) -> float:
        val = cls._safe_float(value, default)
        return max(minimum, min(maximum, val))

    def _refresh_similarity_settings(self) -> None:
        """Update SSIM dedupe thresholds from config/environment."""
        advanced_cfg = self.config.get("advanced") if isinstance(self.config, dict) else {}
        if not isinstance(advanced_cfg, dict):
            advanced_cfg = {}

        threshold_source = os.getenv("AGENT_SSIM_SKIP_THRESHOLD")
        if threshold_source is None:
            threshold_source = advanced_cfg.get("agent_ssim_skip_threshold")
        self.agent_ssim_threshold = self._safe_clamped_float(
            threshold_source,
            self.agent_ssim_threshold,
            0.0,
            1.0
        )

        recheck_source = os.getenv("AGENT_SSIM_RECHECK_SECONDS")
        if recheck_source is None:
            recheck_source = advanced_cfg.get("agent_ssim_recheck_seconds")
        self.agent_ssim_recheck_seconds = max(
            0.0,
            self._safe_float(recheck_source, self.agent_ssim_recheck_seconds)
        )

        cache_ttl_source = os.getenv("AGENT_SSIM_CACHE_TTL_SECONDS")
        if cache_ttl_source is None:
            cache_ttl_source = advanced_cfg.get("agent_ssim_cache_ttl_seconds")
        self.agent_ssim_cache_ttl_seconds = max(
            0.0,
            self._safe_float(cache_ttl_source, self.agent_ssim_cache_ttl_seconds)
        )

        reference_size = advanced_cfg.get("agent_ssim_reference_size")
        if isinstance(reference_size, (list, tuple)) and len(reference_size) == 2:
            try:
                width = int(reference_size[0])
                height = int(reference_size[1])
                if width > 0 and height > 0:
                    self._ssim_reference_size = (width, height)
            except (TypeError, ValueError):
                pass

    def _make_similarity_reference(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Prepare a grayscale reference used for SSIM comparisons."""
        try:
            resized = cv2.resize(frame, self._ssim_reference_size)
            return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        except Exception as exc:
            logger.debug("Failed to prepare similarity reference: %s", exc)
            return None

    def _record_last_event(
        self,
        event: DetectionEvent,
        reference_frame: Optional[np.ndarray],
        analysis_timestamp: float
    ) -> None:
        """Cache the last evaluated event for SSIM-based reuse."""
        self._last_processed_event = copy.deepcopy(event)
        if reference_frame is not None:
            self._last_similarity_frame = reference_frame
        self._last_llm_run_time = analysis_timestamp
        self._last_cache_reset = analysis_timestamp

    def _apply_runtime_config(self) -> None:
        """Reapply runtime configuration values after a config update."""
        camera_config = self.config.get("camera", {}) if isinstance(self.config, dict) else {}

        detection_dir = camera_config.get("detection_image_dir", "detected_images")
        processed_dir = camera_config.get("processed_frames_dir", "processed_frames")

        self.images_dir = Path(detection_dir).expanduser()
        self.processed_frames_dir = Path(processed_dir).expanduser()

        self.save_images = _coerce_bool(camera_config.get("save_detection_images"), False)
        self.save_processed_frames = _coerce_bool(camera_config.get("save_processed_frames"), False)

        if self.save_images:
            try:
                self.images_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.error(f"Failed to ensure detection image directory {self.images_dir}: {exc}")

        if self.save_processed_frames:
            try:
                self.processed_frames_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.error(f"Failed to ensure processed frames directory {self.processed_frames_dir}: {exc}")

        self._refresh_similarity_settings()

    def apply_config(self, updated_config: Dict[str, Any]) -> None:
        """Apply a new configuration payload at runtime."""
        if not isinstance(updated_config, dict):
            raise ValueError("Updated configuration must be a mapping")

        self.config = updated_config
        self.alert_manager.refresh_config(updated_config)
        self._apply_runtime_config()
        self.clear_similarity_cache("config_refresh")

    def clear_similarity_cache(self, reason: str = "manual_reset") -> None:
        """Drop cached SSIM reference data so future frames force fresh analysis."""
        self._last_processed_event = None
        self._last_similarity_frame = None
        self._last_llm_run_time = 0.0
        self._last_cache_reset = time.time()

        log_message = f"SSIM cache cleared ({reason})"
        if reason in {"startup", "config_refresh"}:
            logger.info("♻️ %s", log_message)
        elif reason == "ttl_expired":
            logger.debug(log_message)
        else:
            logger.debug(log_message)

    def _get_rfdet_detector(self) -> Optional[RFDetrPackageDetector]:
        """Lazy-load and cache the RF-DETR package detector."""
        if self._rfdet_detector_failed:
            return None

        if self._rfdet_detector is None:
            try:
                self._rfdet_detector = RFDetrPackageDetector()
            except Exception as exc:  # pragma: no cover - heavy dependency init
                logger.error("Unable to initialize RF-DETR package detector: %s", exc)
                self._rfdet_detector_failed = True
                self._rfdet_detector = None

        return self._rfdet_detector

    def ensure_rfdet_ready(self) -> bool:
        """Ensure the RF-DETR package detector is loaded prior to monitoring."""
        if self._rfdet_detector_failed:
            return False

        detector = self._get_rfdet_detector()
        return detector is not None

    def _save_event_image(
        self,
        image_data: bytes,
        target_dir: Path,
        prefix: str,
        event_time: datetime,
        frame_metadata: Optional[dict] = None
    ) -> Optional[str]:
        """Persist an event image and return the file path."""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error(f"Failed to create image directory {target_dir}: {exc}")
            return None

        timestamp_str = event_time.strftime("%Y%m%d_%H%M%S_%f")
        frame_suffix = ""
        if frame_metadata and frame_metadata.get("frame_number") is not None:
            frame_suffix = f"_f{frame_metadata['frame_number']}"

        filename = f"{prefix}_{timestamp_str}{frame_suffix}.jpg"
        file_path = target_dir / filename

        try:
            with open(file_path, 'wb') as image_file:
                image_file.write(image_data)
            logger.debug(f"Event image saved: {file_path}")
            return str(file_path)
        except Exception as exc:
            logger.error(f"Failed to save event image to {file_path}: {exc}")
            return None
    
    def analyze_frame(self, image_data: bytes, frame_metadata: dict = None) -> Optional[DetectionEvent]:
        """Analyze a frame for packaging box detection using a two-stage LLM approach."""
        try:
            # Log frame processing info
            if frame_metadata:
                similarity_score = frame_metadata.get('similarity_score', 'N/A')
                if isinstance(similarity_score, (int, float)):
                    ssim_str = f"{similarity_score:.3f}"
                else:
                    ssim_str = str(similarity_score)
                
                logger.info(f"🔍 Analyzing frame #{frame_metadata.get('frame_number', 'unknown')} - "
                          f"Reason: {frame_metadata.get('reason', 'unknown')}, "
                          f"SSIM: {ssim_str}, "
                          f"Size: {len(image_data)} bytes")
            else:
                logger.info(f"🔍 Analyzing frame - Size: {len(image_data)} bytes")
            
            analysis_started_at = time.time()
            decoded_frame = None
            reference_frame = None
            similarity_value = None
            gray_frame: Optional[np.ndarray] = None
            hsv_frame: Optional[np.ndarray] = None

            if (
                self.agent_ssim_cache_ttl_seconds > 0.0
                and (analysis_started_at - self._last_cache_reset) >= self.agent_ssim_cache_ttl_seconds
            ):
                logger.info(
                    "♻️ SSIM cache TTL %.1fs elapsed; clearing cached decision before analysis",
                    self.agent_ssim_cache_ttl_seconds
                )
                self.clear_similarity_cache("ttl_expired")

            if image_data:
                try:
                    buffer = np.frombuffer(image_data, dtype=np.uint8)
                    if buffer.size:
                        decoded_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                except Exception as exc:
                    logger.debug("Frame decode for SSIM guard failed: %s", exc)

            if decoded_frame is None:
                logger.warning("Decoded frame is empty; skipping analysis")
                return None

            try:
                gray_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2GRAY)
                hsv_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2HSV)
            except Exception as exc:
                logger.warning("Failed to compute base color spaces: %s", exc)
                return None

            if decoded_frame is not None:
                reference_frame = self._make_similarity_reference(decoded_frame)

            if reference_frame is not None and self._last_similarity_frame is not None:
                try:
                    from skimage.metrics import structural_similarity as ssim

                    similarity_value = ssim(self._last_similarity_frame, reference_frame, data_range=255)
                    time_since_last = analysis_started_at - self._last_llm_run_time
                    if (
                        self._last_processed_event
                        and similarity_value >= self.agent_ssim_threshold
                        and (
                            self.agent_ssim_recheck_seconds <= 0.0
                            or time_since_last <= self.agent_ssim_recheck_seconds
                        )
                    ):
                        logger.info(
                            "♻️ Frame similarity %.3f ≥ threshold %.3f (Δt=%.2fs); reusing previous decision",
                            similarity_value,
                            self.agent_ssim_threshold,
                            time_since_last
                        )
                        reused_event = copy.deepcopy(self._last_processed_event)
                        if reused_event:
                            new_timestamp = datetime.now().isoformat()
                            skip_payload = {
                                "similarity": round(float(similarity_value), 3),
                                "threshold": round(float(self.agent_ssim_threshold), 3),
                                "time_since_last_analysis": round(float(time_since_last), 3),
                                "recheck_window": round(float(self.agent_ssim_recheck_seconds), 3),
                                "count_as_detection": False,
                                "source": "agent_ssim_guard",
                                "previous_primary_label": reused_event.primary_label,
                                "previous_confidence": round(float(reused_event.confidence), 3),
                                "previous_detected": bool(reused_event.detected)
                            }
                            if isinstance(reused_event.decision_trace, dict):
                                reuse_trace = copy.deepcopy(reused_event.decision_trace)
                            else:
                                reuse_trace = {}
                            reuse_trace["agent_similarity_skip"] = skip_payload
                            reuse_trace.setdefault(
                                "classification",
                                reuse_trace.get("classification") or "REUSED_DECISION"
                            )
                            reused_event.decision_trace = reuse_trace
                            reused_event.timestamp = new_timestamp
                            reused_event.should_alert = False
                            if isinstance(reused_event.tools_used, list):
                                reused_event.tools_used = list(reused_event.tools_used)
                            else:
                                reused_event.tools_used = []
                            if isinstance(reused_event.tool_trace, list):
                                reused_event.tool_trace = list(reused_event.tool_trace)
                            else:
                                reused_event.tool_trace = []
                            message_suffix = (
                                f"\n\n[Agent] Reused previous decision due to high frame similarity "
                                f"(SSIM {skip_payload['similarity']:.3f} ≥ {skip_payload['threshold']:.3f})."
                            )
                            reused_event.full_response = (reused_event.full_response or "") + message_suffix
                            return reused_event
                    elif similarity_value is not None and similarity_value >= self.agent_ssim_threshold:
                        logger.debug(
                            "Frame similarity %.3f ≥ threshold %.3f but recheck window elapsed (Δt=%.2fs > %.2fs)",
                            similarity_value,
                            self.agent_ssim_threshold,
                            time_since_last,
                            self.agent_ssim_recheck_seconds
                        )
                except Exception as exc:
                    logger.debug("Agent SSIM comparison failed: %s", exc)

            packaging_analysis = None
            packaging_hint = None
            packaging_box_count = 0
            local_tools_used: List[str] = []
            local_tool_trace: List[Dict[str, Any]] = []
            labels_per_box = 0.0
            label_region_count = 0
            rfdet_analysis = None
            rfdet_hint = None
            if self.packaging_box_analyzer:
                packaging_analysis = self.packaging_box_analyzer.analyze(
                    decoded_frame,
                    gray_frame,
                    hsv_frame
                )
                if packaging_analysis:
                    packaging_box_count = max(0, int(packaging_analysis.estimated_box_count))
                    packaging_hint = packaging_analysis.summary
                    logger.info("📦 Local packaging analysis: %s", packaging_hint)
                packaging_tool_payload = {
                    "detected": bool(packaging_analysis.detected) if packaging_analysis else False,
                    "confidence": float(packaging_analysis.confidence) if packaging_analysis else 0.0,
                    "candidate_count": packaging_analysis.candidate_count if packaging_analysis else 0,
                    "estimated_box_count": packaging_box_count,
                    "summary": packaging_hint or "no analysis"
                }
                local_tools_used.append("packaging_box_analyzer")
                local_tool_trace.append({
                    "name": "packaging_box_analyzer",
                    "output": packaging_tool_payload
                })

            rf_detector = self._get_rfdet_detector()
            if rf_detector:
                rfdet_analysis = rf_detector.analyze(decoded_frame)
                if rfdet_analysis:
                    rfdet_box_count = int(rfdet_analysis.get("box_count", 0) or 0)
                    if rfdet_box_count > packaging_box_count:
                        packaging_box_count = rfdet_box_count
                    rfdet_hint = rfdet_analysis.get("summary")
                    logger.info("🤖 RF-DETR packaging analysis: %s", rfdet_hint)
                    local_tools_used.append("rf_detr_package_detector")
                    local_tool_trace.append({
                        "name": "rf_detr_package_detector",
                        "output": rfdet_analysis
                    })


            if packaging_box_count <= 0:
                labels_per_box = float(label_region_count)
            else:
                labels_per_box = label_region_count / max(packaging_box_count, 1)

            multi_label_flag = packaging_box_count > 0 and labels_per_box > 1.05

            base_prompt = self.config.get("detection", {}).get(
                "prompt",
                "Describe the image in 2-3 concise sentences."
            )
            prompt = base_prompt
            if multi_label_flag:
                prompt = (
                    base_prompt.rstrip() +
                    "\n\nFocus specifically on packaging boxes that appear to have multiple shipping labels. "
                    "Describe each label's placement, orientation, and any distinguishing features if visible."
                )

            # STAGE 1: Vision LLM describes what it sees
            logger.info("🔍 Stage 1: Vision LLM analyzing image...")
            vision_result = self.ollama_client.analyze_image(image_data, prompt)
            vision_description = vision_result.get('response', '')
            
            if len(vision_description) > 500:
                logger.info(f"Vision LLM Response: {vision_description[:500]}... [truncated]")
            else:
                logger.info(f"Vision LLM Response: {vision_description}")
            
            if not vision_description:
                logger.warning("Vision LLM returned empty response")
                return None

            multi_label_description = ""
            if multi_label_flag:
                detailed_prompt = (
                    "Provide a focused description of any boxes that appear to contain more than one shipping label. "
                    "List each box and summarize where the labels sit relative to the box faces."
                )
                try:
                    detailed_result = self.ollama_client.analyze_image(image_data, detailed_prompt)
                    multi_label_description = detailed_result.get('response', '') if isinstance(detailed_result, dict) else ''
                except Exception as detail_exc:
                    logger.warning("Failed to obtain multi-label scene description: %s", detail_exc)
                    multi_label_description = ""

                if multi_label_description:
                    logger.info("📝 Multi-label detail: %s", multi_label_description[:300] + ('...' if len(multi_label_description) > 300 else ''))
                    vision_description = (
                        f"{vision_description}\n\nMulti-label detail:\n{multi_label_description}"
                    )
                    local_tools_used.append("vision_multi_label_detail")
                    local_tool_trace.append({
                        "name": "vision_multi_label_detail",
                        "output": {
                            "prompt": detailed_prompt,
                            "response": multi_label_description
                        }
                    })
            
            # STAGE 2: Decision LLM determines if an unlabeled packaging box needs attention
            logger.info("🤖 Stage 2: Decision LLM evaluating...")
            decision_result = self.decision_llm.make_detection_decision(vision_description, None)
            if not isinstance(decision_result, dict):
                decision_result = {}

            raw_trace = decision_result.get("decision_trace", {})
            decision_trace = raw_trace.copy() if isinstance(raw_trace, dict) else {}

            if packaging_analysis:
                decision_trace.setdefault("packaging_analysis", packaging_analysis.to_dict())
            elif packaging_hint and "packaging_analysis" not in decision_trace:
                decision_trace["packaging_analysis"] = {"summary": packaging_hint}

            if rfdet_analysis:
                decision_trace.setdefault("rf_detr_analysis", rfdet_analysis)
            elif rfdet_hint and "rf_detr_analysis" not in decision_trace:
                decision_trace["rf_detr_analysis"] = {"summary": rfdet_hint}

            tools_used_llm = decision_trace.get("tools_used")
            if isinstance(tools_used_llm, list):
                tools_used_llm = [str(tool) for tool in tools_used_llm]
            elif tools_used_llm is None:
                tools_used_llm = []
            else:
                tools_used_llm = [str(tools_used_llm)]

            tool_trace_llm = decision_trace.get("tool_trace")
            if not isinstance(tool_trace_llm, list):
                tool_trace_llm = []

            combined_tool_trace = local_tool_trace + tool_trace_llm
            combined_tools_used: List[str] = []
            for name in local_tools_used + tools_used_llm:
                if name and name not in combined_tools_used:
                    combined_tools_used.append(name)

            decision_trace["packaging_box_count"] = packaging_box_count
            decision_trace["label_region_count"] = label_region_count
            decision_trace["labels_per_box"] = round(float(labels_per_box), 3)
            decision_trace["multi_label_flag"] = bool(multi_label_flag)
            if multi_label_flag and multi_label_description and "multi_label_description" not in decision_trace:
                decision_trace["multi_label_description"] = multi_label_description

            packaging_confidence_value = packaging_analysis.confidence if packaging_analysis else None

            if packaging_confidence_value is not None:
                decision_trace.setdefault("packaging_confidence", round(float(packaging_confidence_value), 3))

            decision_trace["tools_used"] = combined_tools_used
            decision_trace["tool_trace"] = combined_tool_trace

            if similarity_value is not None:
                decision_trace.setdefault("agent_similarity_snapshot", {
                    "similarity": round(float(similarity_value), 3),
                    "threshold": round(float(self.agent_ssim_threshold), 3),
                    "recheck_window": round(float(self.agent_ssim_recheck_seconds), 3)
                })

            llm_scene_metrics = decision_trace.get("scene_metrics")
            if isinstance(llm_scene_metrics, dict):
                llm_box_count = llm_scene_metrics.get("box_count")
                llm_label_count = llm_scene_metrics.get("label_count")
            else:
                llm_box_count = None
                llm_label_count = None

            llm_box_count = decision_trace.get("llm_reported_box_count", llm_box_count)
            llm_label_count = decision_trace.get("llm_reported_label_count", llm_label_count)

            report_box_count = llm_box_count if llm_box_count is not None else packaging_box_count
            report_label_count = llm_label_count if llm_label_count is not None else label_region_count
            if report_box_count is None:
                report_box_count = 0
            if report_label_count is None:
                report_label_count = 0

            if report_box_count > 0:
                report_labels_per_box = report_label_count / max(report_box_count, 1)
            else:
                report_labels_per_box = float(report_label_count)

            labels_per_box = report_labels_per_box
            packaging_box_count = report_box_count
            label_region_count = report_label_count

            decision_trace["final_box_count"] = report_box_count
            decision_trace["final_label_count"] = report_label_count
            decision_trace["labels_per_box"] = round(float(report_labels_per_box), 3)

            if isinstance(decision_result, dict):
                decision_result["decision_trace"] = decision_trace
                decision_result["tools_used"] = combined_tools_used
                decision_result["tool_trace"] = combined_tool_trace

            classification_snapshot = decision_trace.get("classification", "")
            tools_summary = ", ".join(combined_tools_used) if combined_tools_used else "none"
            logger.info("🧾 Decision trace summary: classification=%s, tools=%s", classification_snapshot or "unknown", tools_summary)
            if decision_trace:
                try:
                    logger.debug("Decision trace detail: %s", json.dumps(decision_trace, indent=2, default=str))
                except TypeError:
                    logger.debug("Decision trace detail (non-serializable) recorded")

            analysis_completed_at = time.time()

            def blend_confidence(base_conf: float, shipping_label_present: Optional[bool]) -> Tuple[float, Dict[str, float]]:
                base_conf = max(0.0, min(1.0, float(base_conf or 0.0)))
                components: Dict[str, float] = {
                    "llm": round(base_conf, 3),
                    "final_confidence": round(base_conf, 3)
                }
                return base_conf, components

            detected = bool(decision_result.get("detected"))
            event_time = datetime.now()
            timestamp_iso = event_time.isoformat()

            if detected:
                confidence = float(decision_result.get("confidence", 0.85))
                reasoning = decision_result.get("reasoning", "Packaging box identified by decision LLM")
                shipping_label_present = decision_result.get("shipping_label_present")
                if isinstance(shipping_label_present, str):
                    shipping_label_present = shipping_label_present.strip().lower() in {"true", "1", "yes"}

                should_alert = decision_result.get("should_alert")
                if isinstance(should_alert, str):
                    should_alert = should_alert.strip().lower() in {"true", "1", "yes"}
                if should_alert is None:
                    should_alert = not bool(shipping_label_present)

                final_confidence, confidence_components = blend_confidence(confidence, shipping_label_present)
                decision_trace["llm_reported_confidence"] = round(confidence, 3)
                decision_trace["confidence_blend"] = confidence_components
                confidence = final_confidence
                decision_result["confidence"] = confidence

                primary_label = "packaging_box_unlabeled" if not shipping_label_present else "packaging_box_with_label"

                if shipping_label_present:
                    logger.info("📦 Packaging box detected with shipping label present; logging without alert")
                else:
                    logger.info(f"🚨 UNLABELED PACKAGING BOX DETECTED! Confidence: {confidence:.2f}")
                logger.info(f"🎯 Detection reasoning: {reasoning}")

                label_status = "visible" if shipping_label_present else "not visible"
                label_status = "unknown" if shipping_label_present is None else label_status

                if packaging_analysis:
                    decision_trace.setdefault("packaging_analysis", packaging_analysis.to_dict())

                combined_response = (
                    "Vision description:\n"
                    f"{vision_description}\n\n"
                    "Decision reasoning:\n"
                    f"{reasoning}\n\n"
                    f"Shipping label status: {label_status}"
                )

                if packaging_hint:
                    combined_response += f"\n\nLocal packaging analysis: {packaging_hint}"
                if rfdet_hint:
                    combined_response += f"\n\nRF-DETR packaging analysis: {rfdet_hint}"
                combined_response += (
                    f"\n\nScene counts: boxes≈{packaging_box_count}, label_regions≈{label_region_count}, "
                    f"labels_per_box≈{round(float(labels_per_box), 3)}"
                )
                combined_response += f"\n\nDecision tools: {', '.join(combined_tools_used) if combined_tools_used else 'none'}"

                detection_event = DetectionEvent(
                    timestamp=timestamp_iso,
                    confidence=confidence,
                    primary_label=primary_label,
                    full_response=combined_response,
                    image_path=None,
                    vision_description=vision_description,
                    decision_trace=decision_trace,
                    detected=True,
                    shipping_label_present=shipping_label_present,
                    should_alert=bool(should_alert),
                    tools_used=combined_tools_used,
                    tool_trace=combined_tool_trace
                )

                if self.save_images:
                    image_path = self._save_event_image(
                        image_data,
                        self.images_dir,
                        "detection",
                        event_time,
                        frame_metadata
                    )
                    if image_path:
                        detection_event.image_path = image_path
                self._record_last_event(detection_event, reference_frame, analysis_completed_at)
                return detection_event

            # No detection – record outcome for visibility
            classification_text = decision_trace.get("classification") or decision_result.get("classification")
            if not classification_text:
                classification_text = "NO_BOX_DETECTED"

            base_conf_no_detection = float(decision_result.get("confidence", 0.0) or 0.0)
            final_conf_no_detection, components_no_detection = blend_confidence(
                base_conf_no_detection,
                decision_result.get("shipping_label_present")
            )
            decision_trace.setdefault("llm_reported_confidence", round(base_conf_no_detection, 3))
            decision_trace.setdefault("confidence_blend", components_no_detection)
            decision_result["confidence"] = final_conf_no_detection

            combined_response = (
                "Vision description:\n"
                f"{vision_description}\n\n"
                "Decision outcome:\n"
                f"{classification_text}"
            )

            if packaging_hint:
                combined_response += f"\n\nLocal packaging analysis: {packaging_hint}"
            if rfdet_hint:
                combined_response += f"\n\nRF-DETR packaging analysis: {rfdet_hint}"
            combined_response += (
                f"\n\nScene counts: boxes≈{packaging_box_count}, label_regions≈{label_region_count}, "
                f"labels_per_box≈{round(float(labels_per_box), 3)}"
            )
            combined_response += f"\n\nDecision tools: {', '.join(combined_tools_used) if combined_tools_used else 'none'}"

            logger.info("✅ No packaging box requiring action detected by decision LLM")

            detection_event = DetectionEvent(
                timestamp=timestamp_iso,
                confidence=final_conf_no_detection,
                primary_label="no_detection",
                full_response=combined_response,
                image_path=None,
                vision_description=vision_description,
                decision_trace=decision_trace,
                detected=False,
                shipping_label_present=None,
                should_alert=False,
                tools_used=combined_tools_used,
                tool_trace=combined_tool_trace
            )

            if self.save_processed_frames:
                image_path = self._save_event_image(
                    image_data,
                    self.processed_frames_dir,
                    "event",
                    event_time,
                    frame_metadata
                )
                if image_path:
                    detection_event.image_path = image_path

            self._record_last_event(detection_event, reference_frame, analysis_completed_at)
            return detection_event
                
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}", exc_info=True)
            self.last_error = str(e)
            return None
    
    def process_detection(self, event: DetectionEvent) -> bool:
        """Process a detection event and send alerts."""
        if not event.detected:
            logger.debug("Skipping alert processing for non-detection event")
            return False

        if event.shipping_label_present:
            logger.info("Skipping alert: shipping label present on detected packaging box")
            return False

        if not event.should_alert:
            logger.info("Skipping alert per decision model guidance")
            return False

        logger.info(f"Processing detection: {event.primary_label} (confidence: {event.confidence:.2f})")
        alerts_sent = 0
        
        # Process each rule
        for rule in self.config.get("rules", []):
            if not rule.get("enabled", True):
                continue
            
            # Send email alert
            if self.alert_manager.send_email_alert(event, rule):
                alerts_sent += 1
        
        # Send desktop notification if enabled
        if self.config.get("notifications", {}).get("desktop", {}).get("enabled", False):
            if self.alert_manager.send_desktop_notification(event):
                alerts_sent += 1
        return alerts_sent > 0
    