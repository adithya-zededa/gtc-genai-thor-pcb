"""Classical detection utilities for packaging box analysis."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from agent_runtime.utils import clamp_float


logger = logging.getLogger(__name__)
Array = NDArray[Any]


DEFAULT_PACKAGING_ANALYZER_PARAMS: Dict[str, Any] = {
    "min_area_ratio": 0.004,
    "max_area_ratio": 0.7,
    "aspect_range": (0.3, 4.2),
    "min_rectangularity": 0.5,
    "min_solidity": 0.55,
    "min_vertices": 4,
    "max_vertices": 10,
    "min_color_area_ratio": 0.002,
    "min_edge_density": 0.02,
    "target_aspect": 1.6,
    "score_weights": {
        "area": 0.25,
        "rectangularity": 0.25,
        "solidity": 0.2,
        "aspect": 0.15,
        "color": 0.15,
    },
    "color_mask_bonus": 0.07,
    "edge_density_scale": 2.0,
    "brown_hue_range": (5, 40),
    "brown_saturation_min": 45.0,
    "brown_value_range": (50.0, 225.0),
    "brown_hue_center": 22.5,
    "brown_hue_span": 17.5,
    "brown_sat_scale": 140.0,
    "brown_val_scale": 175.0,
    "gaussian_kernel": (5, 5),
    "canny_thresholds": (35, 120),
    "morph_kernel": (5, 5),
    "morph_iterations": 2,
    "dilate_iterations": 1,
    "approx_poly_factor": 0.035,
    "color_kernel": (9, 9),
    "color_iterations": 2,
    "color_dilate_iterations": 1,
    "color_ranges": [
        {"lower": [5, 60, 40], "upper": [25, 180, 200]},
        {"lower": [10, 50, 60], "upper": [30, 180, 230]},
    ],
    "strong_score_threshold": 0.58,
    "moderate_score_threshold": 0.45,
    "base_detection_threshold": 0.45,
    "density_scale": 2.5,
    "confidence_weights": {
        "top_score": 0.5,
        "avg_top": 0.25,
        "density": 0.15,
        "color_density": 0.1,
    },
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
    candidates: List[Dict[str, Any]] = field(default_factory=lambda: [])
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


class PackagingBoxAnalyzer:
    """Classical vision pipeline to flag packaging-box shaped regions."""

    def __init__(self, **params: Any) -> None:
        cfg = copy.deepcopy(DEFAULT_PACKAGING_ANALYZER_PARAMS)
        for key, value in params.items():
            if value is not None:
                cfg[key] = value

        self.min_area_ratio = float(cfg["min_area_ratio"])
        self.max_area_ratio = float(cfg["max_area_ratio"])
        self.aspect_range = tuple(cfg["aspect_range"])
        self.min_rectangularity = float(cfg["min_rectangularity"])
        self.min_solidity = float(cfg["min_solidity"])
        self.min_vertices = int(cfg["min_vertices"])
        self.max_vertices = int(cfg["max_vertices"])
        self.min_color_area_ratio = float(cfg["min_color_area_ratio"])
        self.min_edge_density = float(cfg["min_edge_density"])
        self.target_aspect = float(cfg["target_aspect"])
        default_score_weights = DEFAULT_PACKAGING_ANALYZER_PARAMS["score_weights"]
        user_score_weights = cfg.get("score_weights", {})
        if not isinstance(user_score_weights, dict):
            user_score_weights = {}
        merged_score_weights: Dict[str, Any] = {
            **default_score_weights,
            **user_score_weights,
        }
        self.score_weights: Dict[str, float] = {
            key: float(value) for key, value in merged_score_weights.items()
        }
        self.color_mask_bonus = float(cfg["color_mask_bonus"])
        self.edge_density_scale = float(cfg["edge_density_scale"])
        self.brown_hue_range = tuple(cfg["brown_hue_range"])
        self.brown_saturation_min = float(cfg["brown_saturation_min"])
        self.brown_value_range = tuple(cfg["brown_value_range"])
        self.brown_hue_center = float(cfg["brown_hue_center"])
        self.brown_hue_span = float(cfg["brown_hue_span"])
        self.brown_sat_scale = float(cfg["brown_sat_scale"])
        self.brown_val_scale = float(cfg["brown_val_scale"])
        self.gaussian_kernel = tuple(cfg["gaussian_kernel"])
        self.canny_thresholds = tuple(cfg["canny_thresholds"])
        self.morph_kernel = tuple(cfg["morph_kernel"])
        self.morph_iterations = int(cfg["morph_iterations"])
        self.dilate_iterations = int(cfg["dilate_iterations"])
        self.approx_poly_factor = float(cfg["approx_poly_factor"])
        self.color_kernel = tuple(cfg["color_kernel"])
        self.color_iterations = int(cfg["color_iterations"])
        self.color_dilate_iterations = int(cfg["color_dilate_iterations"])
        self.color_ranges = cfg.get("color_ranges", [])
        self.strong_score_threshold = float(cfg["strong_score_threshold"])
        self.moderate_score_threshold = float(cfg["moderate_score_threshold"])
        self.base_detection_threshold = float(cfg["base_detection_threshold"])
        self.density_scale = float(cfg["density_scale"])
        self.confidence_weights = {
            key: float(value) for key, value in cfg["confidence_weights"].items()
        }

    def analyze(
        self,
        frame_bgr: Optional[Array],
        gray_frame: Optional[Array],
        hsv_frame: Optional[Array],
    ) -> Optional[PackagingBoxAnalysis]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            eps = 1e-6
            height, width = frame_bgr.shape[:2]
            frame_area = float(height * width)

            # Edge detection preprocessing
            blurred = cv2.GaussianBlur(frame_bgr, self.gaussian_kernel, 0)
            edges = cv2.Canny(blurred, *self.canny_thresholds)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel, iterations=self.morph_iterations)
            edges = cv2.dilate(edges, morph_kernel, iterations=self.dilate_iterations)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: List[Dict[str, Any]] = []

            for contour in contours:
                if contour is None or len(contour) < self.min_vertices:
                    continue

                area = float(cv2.contourArea(contour))
                if area <= 0:
                    continue
                area_ratio = area / frame_area
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    continue

                perimeter = float(cv2.arcLength(contour, True))
                rect_area = float(perimeter * perimeter / (16 * np.pi))
                rectangularity = area / max(rect_area, eps)
                if rectangularity < self.min_rectangularity:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / max(hull_area, eps)
                if solidity < self.min_solidity:
                    continue

                x, y, w_box, h_box = cv2.boundingRect(contour)
                aspect_ratio = w_box / max(h_box, 1)
                if aspect_ratio < min(self.aspect_range) or aspect_ratio > max(self.aspect_range):
                    continue

                roi_edges = edges[y : y + h_box, x : x + w_box]
                edge_density = float(np.count_nonzero(roi_edges)) / max(area, eps)
                if edge_density < self.min_edge_density:
                    continue

                mask = np.zeros_like(gray_frame)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
                mean_bgr = cv2.mean(masked, mask=mask)[:3]
                mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

                score_components = self._score_candidate(
                    area_ratio=area_ratio,
                    aspect_ratio=aspect_ratio,
                    rectangularity=rectangularity,
                    solidity=solidity,
                    edge_density=edge_density,
                    mean_hsv=mean_hsv,
                )

                candidate_payload = {
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
                    "source": "edges",
                }
                candidates.append(candidate_payload)

            # Color masks for brown cardboard heuristics
            if hsv_frame is not None:
                color_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.color_kernel)
                combined_mask = np.zeros_like(gray_frame)
                for range_cfg in self.color_ranges:
                    lower = np.array(range_cfg.get("lower", [0, 0, 0]), dtype=np.uint8)
                    upper = np.array(range_cfg.get("upper", [180, 255, 255]), dtype=np.uint8)
                    mask = cv2.inRange(hsv_frame, lower, upper)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color_kernel, iterations=self.color_iterations)
                    mask = cv2.dilate(mask, color_kernel, iterations=self.color_dilate_iterations)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

                color_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in color_contours:
                    if contour is None or len(contour) < self.min_vertices:
                        continue

                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    area = float(cv2.contourArea(contour))
                    area_ratio = area / max(frame_area, 1.0)
                    if area_ratio < self.min_color_area_ratio:
                        continue

                    aspect_ratio = w_box / max(h_box, 1)
                    rectangularity = area / max(w_box * h_box, 1.0)
                    hull = cv2.convexHull(contour)
                    hull_area = float(cv2.contourArea(hull))
                    solidity = area / max(hull_area, 1.0)
                    edge_density = 0.0
                    mean_hsv = cv2.mean(hsv_frame[y : y + h_box, x : x + w_box])[:3]
                    score_components = self._score_candidate(
                        area_ratio=area_ratio,
                        aspect_ratio=aspect_ratio,
                        rectangularity=rectangularity,
                        solidity=solidity,
                        edge_density=edge_density,
                        mean_hsv=mean_hsv,
                        source="color_mask",
                    )
                    candidates.append(
                        {
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
                            "source": "color_mask",
                        }
                    )

            candidate_count = len(candidates)
            if candidate_count:
                avg_aspect = sum(c["aspect_ratio"] for c in candidates) / candidate_count
                avg_rectangularity = sum(c["rectangularity"] for c in candidates) / candidate_count
                avg_solidity = sum(c["solidity"] for c in candidates) / candidate_count
                top_score = max(c["score"] for c in candidates)
                top_scores = sorted((c["score"] for c in candidates), reverse=True)[:3]
                avg_top = sum(top_scores) / len(top_scores)
                density = clamp_float(candidate_count / max(self.density_scale, eps), 0.0, 1.0)
                color_density = clamp_float(
                    sum(1 for c in candidates if c.get("source") == "color_mask") / max(candidate_count, 1),
                    0.0,
                    1.0,
                )
                confidence = clamp_float(
                    self.confidence_weights["top_score"] * top_score
                    + self.confidence_weights["avg_top"] * avg_top
                    + self.confidence_weights["density"] * density
                    + self.confidence_weights["color_density"] * color_density,
                    0.0,
                    1.0,
                )

                strong_candidates = [
                    c for c in candidates if c.get("score", 0.0) >= self.strong_score_threshold
                ]
                moderate_candidates = [
                    c for c in candidates if c.get("score", 0.0) >= self.moderate_score_threshold
                ]
                if strong_candidates:
                    estimated_box_count = len(strong_candidates)
                elif moderate_candidates:
                    estimated_box_count = len(moderate_candidates)
                else:
                    estimated_box_count = 1 if top_score >= self.moderate_score_threshold else 0
            else:
                avg_aspect = 0.0
                avg_rectangularity = 0.0
                avg_solidity = 0.0
                confidence = 0.1
                top_score = 0.0
                estimated_box_count = 0

            detected = candidate_count > 0 and confidence >= self.base_detection_threshold
            summary = (
                f"{candidate_count} packaging-like region{'s' if candidate_count != 1 else ''} "
                f"(confidence {confidence:.2f}, top score {top_score:.2f}, estimated boxes {estimated_box_count})"
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
                estimated_box_count=estimated_box_count,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Packaging analyzer failed: %s", exc, exc_info=True)
            return None

    def _score_candidate(
        self,
        *,
        area_ratio: float,
        aspect_ratio: float,
        rectangularity: float,
        solidity: float,
        edge_density: float,
        mean_hsv: Tuple[float, float, float],
        source: str = "edges",
    ) -> Dict[str, float]:
        score = 0.0
        area_component = clamp_float(area_ratio / max(self.max_area_ratio, 1e-6), 0.0, 1.0)
        aspect_deviation = abs(aspect_ratio - self.target_aspect)
        aspect_component = clamp_float(1.0 - aspect_deviation / self.target_aspect, 0.0, 1.0)
        rectangularity_component = clamp_float(rectangularity, 0.0, 1.0)
        solidity_component = clamp_float(solidity, 0.0, 1.0)
        edge_component = clamp_float(edge_density * self.edge_density_scale, 0.0, 1.0)

        hue, saturation, value = mean_hsv
        hue_distance = abs(hue - self.brown_hue_center)
        hue_component = clamp_float(1.0 - hue_distance / max(self.brown_hue_span, 1e-6), 0.0, 1.0)
        saturation_component = clamp_float(saturation / self.brown_sat_scale, 0.0, 1.0)
        value_component = clamp_float(value / self.brown_val_scale, 0.0, 1.0)
        color_component = (hue_component + saturation_component + value_component) / 3.0

        score += self.score_weights.get("area", 0.0) * area_component
        score += self.score_weights.get("rectangularity", 0.0) * rectangularity_component
        score += self.score_weights.get("solidity", 0.0) * solidity_component
        score += self.score_weights.get("aspect", 0.0) * aspect_component
        score += self.score_weights.get("color", 0.0) * color_component

        if source == "color_mask":
            score += self.color_mask_bonus

        return {
            "base": clamp_float(score, 0.0, 1.0),
            "area": round(area_component, 4),
            "aspect": round(aspect_component, 4),
            "rectangularity": round(rectangularity_component, 4),
            "solidity": round(solidity_component, 4),
            "color": round(color_component, 4),
            "edge": round(edge_component, 4),
        }
