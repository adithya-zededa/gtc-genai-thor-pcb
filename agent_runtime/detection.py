"""Classical detection utilities and RF-DETR integration."""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from agent_runtime.utils import clamp_float

try:  # pragma: no cover - optional dependency
    from skimage.metrics import structural_similarity  # type: ignore
except ImportError:  # pragma: no cover - best effort fallback
    structural_similarity = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from rfdetr.detr import RFDETRMedium  # type: ignore
except ImportError:  # pragma: no cover - best effort fallback
    RFDETRMedium = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover - huggingface optional

    def hf_hub_download(*args: Any, **kwargs: Any) -> str:  # type: ignore
        raise RuntimeError("huggingface_hub.hf_hub_download not available")


logger = logging.getLogger(__name__)
Array = NDArray[Any]


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
                frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
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
            "area": area_component,
            "aspect": aspect_component,
            "rectangularity": rectangularity_component,
            "solidity": solidity_component,
            "color": color_component,
            "edge": edge_component,
        }


DEFAULT_RFDETR_CONFIG: Dict[str, Any] = {
    "threshold": 0.4,
    "repo_id": "Mact0/rf-detr-package-detection",
    "checkpoint_filename": "checkpoint_best_total.pth",
}


class RFDetrPackageDetector:
    """Wrapper around the RF-DETR Medium package detector from Hugging Face."""

    def __init__(self, **params: Any) -> None:
        cfg = DEFAULT_RFDETR_CONFIG.copy()
        for key, value in params.items():
            if value is not None:
                cfg[key] = value

        self.threshold = float(cfg.get("threshold", DEFAULT_RFDETR_CONFIG["threshold"]))
        self.repo_id = str(cfg.get("repo_id", DEFAULT_RFDETR_CONFIG["repo_id"]))
        self.checkpoint_filename = str(cfg.get("checkpoint_filename", DEFAULT_RFDETR_CONFIG["checkpoint_filename"]))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            if RFDETRMedium is None:
                raise ImportError("RFDETRMedium is not available; install rfdetr package")
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.checkpoint_filename,
                    repo_type="model",
                )

                model = RFDETRMedium()
                model.model.reinitialize_detection_head(num_classes=2)
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

    def analyze(self, frame: Optional[Array]) -> Optional[Dict[str, Any]]:
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
                    "summary": "RF-DETR detected no packaging boxes",
                }

            boxes_list: List[Dict[str, Any]] = []
            scores: List[float] = []
            if boxes is None:
                boxes_list = []
                scores = []
            else:
                for idx in range(count):
                    x1, y1, x2, y2 = [float(val) for val in boxes[idx]]
                    score = float(confidences[idx]) if confidences is not None else 0.0
                    cls_id = int(labels[idx]) if labels is not None else 0
                    boxes_list.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "score": round(score, 4),
                            "class_id": cls_id,
                        }
                    )
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
                "summary": summary,
            }
        except Exception as exc:  # pragma: no cover - best effort logging path
            logger.error("RF-DETR inference failed: %s", exc, exc_info=True)
            return None


DEFAULT_SHIPPING_ANALYZER_PARAMS: Dict[str, Any] = {
    "min_area_ratio": 0.001,
    "max_area_ratio": 0.25,
    "min_fill_ratio": 0.5,
    "aspect_range": (0.45, 6.0),
    "color_min_area_ratio": 0.0006,
    "yellow_hue_range": (15, 45),
    "white_value_threshold": 200,
    "white_saturation_max": 40,
    "min_color_region_area": 180,
    "max_cluster_candidates": 6,
    "color_bonus": 0.08,
    "yellow_sat_threshold": 60,
    "yellow_val_threshold": 110,
}


class ShippingLabelAnalyzer:
    """Lightweight CV + clustering model to identify shipping label regions."""

    def __init__(self, **params: Any) -> None:
        cfg = DEFAULT_SHIPPING_ANALYZER_PARAMS.copy()
        for key, value in params.items():
            if value is not None:
                cfg[key] = value

        self.min_area_ratio = float(cfg["min_area_ratio"])
        self.max_area_ratio = float(cfg["max_area_ratio"])
        self.min_fill_ratio = float(cfg["min_fill_ratio"])
        self.aspect_range = tuple(cfg["aspect_range"])
        self.color_min_area_ratio = float(cfg["color_min_area_ratio"])
        self.yellow_hue_range = tuple(cfg["yellow_hue_range"])
        self.white_value_threshold = float(cfg["white_value_threshold"])
        self.white_saturation_max = float(cfg["white_saturation_max"])
        self.min_color_region_area = float(cfg["min_color_region_area"])
        self.max_cluster_candidates = int(cfg["max_cluster_candidates"])
        self.color_bonus = float(cfg["color_bonus"])
        self.yellow_sat_threshold = float(cfg["yellow_sat_threshold"])
        self.yellow_val_threshold = float(cfg["yellow_val_threshold"])

    def _cluster_candidates(
        self, candidates: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        if not candidates:
            return {"region_count": 0, "cluster_confidences": [], "labels": None, "method": "kmeans"}

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
            norm_area = float(area_ratio) * 8.0
            features.append([norm_cx, norm_cy, norm_area])

        data = np.array(features, dtype=np.float32)
        if data.shape[0] == 1:
            return {
                "region_count": 1,
                "cluster_confidences": [candidates[0].get("score", 0.5)],
                "labels": np.array([0], dtype=np.int32),
                "method": "kmeans",
            }

        max_k = min(self.max_cluster_candidates, data.shape[0])
        best_k = 1
        best_penalty = float("inf")
        best_result = {"labels": np.zeros((data.shape[0], 1), dtype=np.int32), "center": None, "compactness": 0.0}

        for k in range(1, max_k + 1):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
            try:
                compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            except Exception:
                continue

            penalty = compactness + (k**1.5) * 0.08
            if penalty < best_penalty:
                best_penalty = penalty
                best_k = k
                best_result = {"labels": labels, "centers": centers, "compactness": compactness}

        labels = best_result.get("labels")
        if labels is None:
            labels = np.zeros((data.shape[0], 1), dtype=np.int32)

        cluster_confidences: List[float] = []
        for cluster_id in range(best_k):
            cluster_members = [candidates[idx] for idx, label in enumerate(labels.flatten()) if label == cluster_id]
            if not cluster_members:
                cluster_confidences.append(0.0)
                continue
            top_score = max(member.get("score", 0.0) for member in cluster_members)
            mean_score = sum(member.get("score", 0.0) for member in cluster_members) / len(cluster_members)
            cluster_confidences.append(clamp_float(0.65 * top_score + 0.35 * mean_score, 0.0, 1.0))

        cluster_confidences.sort(reverse=True)
        return {"region_count": best_k, "cluster_confidences": cluster_confidences, "labels": labels, "method": "kmeans"}

    def _score_candidate(
        self,
        *,
        area_ratio: float,
        aspect_ratio: float,
        fill_ratio: float,
        mean_intensity: float,
        mean_hsv: Optional[List[float]],
        source: str = "contour",
    ) -> Dict[str, float]:
        eps = 1e-6
        aspect_score = 0.0
        if aspect_ratio > eps:
            aspect_score = np.exp(-abs(np.log(aspect_ratio)))
            aspect_score = clamp_float(aspect_score, 0.0, 1.0)

        fill_score = clamp_float(fill_ratio, 0.0, 1.0)
        area_score = clamp_float((area_ratio - self.min_area_ratio) / max(self.max_area_ratio - self.min_area_ratio, eps), 0.0, 1.0)
        brightness_score = clamp_float(mean_intensity / 200.0, 0.0, 1.0)

        hue = sat = val = 0.0
        if mean_hsv and len(mean_hsv) == 3:
            hue, sat, val = mean_hsv

        yellow_score = 0.0
        if self.yellow_hue_range[0] <= hue <= self.yellow_hue_range[1] and sat >= self.yellow_sat_threshold and val >= self.yellow_val_threshold:
            yellow_score = clamp_float((sat - self.yellow_sat_threshold) / 140.0, 0.0, 1.0) * 0.6 + clamp_float((val - self.yellow_val_threshold) / 145.0, 0.0, 1.0) * 0.4

        white_score = 0.0
        if val >= self.white_value_threshold and sat <= self.white_saturation_max + 20:
            white_score = clamp_float((val - self.white_value_threshold) / 55.0, 0.0, 1.0)

        color_score = max(yellow_score, white_score)
        combined_color = max(color_score, brightness_score)
        base_score = 0.3 * area_score + 0.25 * aspect_score + 0.3 * fill_score + 0.15 * combined_color

        if source == "color_mask":
            base_score = clamp_float(base_score + self.color_bonus, 0.0, 1.0)

        return {
            "area": round(area_score, 3),
            "aspect": round(aspect_score, 3),
            "fill": round(fill_score, 3),
            "color": round(combined_color, 3),
            "base": round(clamp_float(base_score, 0.0, 1.0), 3),
        }

    def analyze(self, frame: Optional[Array], gray: Optional[Array], hsv: Optional[Array]) -> Optional[ShippingLabelAnalysis]:
        try:
            if frame is None or gray is None or hsv is None:
                return None
            if gray.ndim != 2 or hsv.ndim != 3:
                return None

            height, width = gray.shape[:2]
            if height == 0 or width == 0:
                return None

            image_area = float(width * height)
            eps = 1e-6

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

                region = gray[y : y + h_box, x : x + w_box]
                mean_intensity = float(region.mean()) if region.size else 0.0
                region_hsv = hsv[y : y + h_box, x : x + w_box]
                mean_hsv = [float(region_hsv[:, :, idx].mean()) if region_hsv.size else 0.0 for idx in range(3)]
                score_components = self._score_candidate(
                    area_ratio=area_ratio, aspect_ratio=aspect_ratio, fill_ratio=fill_ratio, mean_intensity=mean_intensity, mean_hsv=mean_hsv, source="contour"
                )

                candidates.append(
                    {
                        "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                        "aspect_ratio": round(aspect_ratio, 3),
                        "fill_ratio": round(fill_ratio, 3),
                        "mean_intensity": round(mean_intensity, 1),
                        "area_ratio": round(area_ratio, 4),
                        "area_ratio_raw": float(area_ratio),
                        "mean_hsv": [round(channel, 1) for channel in mean_hsv],
                        "source": "contour",
                        "score": score_components["base"],
                        "score_components": score_components,
                    }
                )

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
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
                        solidity = area / hull_area if hull_area else 0.0
                        if solidity < self.min_fill_ratio:
                            continue

                        color_region = hsv[y : y + h_box, x : x + w_box]
                        mean_hue = float(color_region[:, :, 0].mean()) if color_region.size else 0.0
                        mean_sat = float(color_region[:, :, 1].mean()) if color_region.size else 0.0
                        mean_val = float(color_region[:, :, 2].mean()) if color_region.size else 0.0

                        region_gray = gray[y : y + h_box, x : x + w_box]
                        mean_intensity = float(region_gray.mean()) if region_gray.size else 0.0
                        score_components = self._score_candidate(
                            area_ratio=area_ratio, aspect_ratio=aspect_ratio, fill_ratio=solidity, mean_intensity=mean_intensity, mean_hsv=[mean_hue, mean_sat, mean_val], source="color_mask"
                        )

                        candidates.append(
                            {
                                "bounding_box": [int(x), int(y), int(w_box), int(h_box)],
                                "aspect_ratio": round(aspect_ratio, 3),
                                "fill_ratio": round(solidity, 3),
                                "mean_hsv": [round(mean_hue, 1), round(mean_sat, 1), round(mean_val, 1)],
                                "mean_intensity": round(mean_intensity, 1),
                                "area_ratio": round(area_ratio, 4),
                                "area_ratio_raw": float(area_ratio),
                                "source": "color_mask",
                                "score": score_components["base"],
                                "score_components": score_components,
                            }
                        )

            candidate_count = len(candidates)
            if candidate_count:
                avg_aspect = sum(c.get("aspect_ratio", 0.0) for c in candidates) / candidate_count
                cluster_summary = self._cluster_candidates(candidates, (height, width))
                region_count = cluster_summary.get("region_count", 0)
                cluster_confidences = cluster_summary.get("cluster_confidences", [])
                max_cluster_conf = cluster_confidences[0] if cluster_confidences else 0.0
                avg_top_clusters = sum(cluster_confidences[:3]) / max(1, min(3, len(cluster_confidences))) if cluster_confidences else 0.0
                density_score = clamp_float(region_count / 4.0, 0.0, 1.0)
                color_hits = sum(1 for c in candidates if c.get("source") == "color_mask")
                color_density = clamp_float(color_hits / candidate_count, 0.0, 1.0) if candidate_count else 0.0
                presence_confidence = clamp_float(0.5 * max_cluster_conf + 0.25 * avg_top_clusters + 0.15 * density_score + 0.1 * color_density, 0.0, 1.0)
            else:
                avg_aspect = 0.0
                presence_confidence = 0.02
                cluster_summary = {"region_count": 0, "cluster_confidences": [], "method": "kmeans"}
                max_cluster_conf = 0.0
                region_count = 0

            detected = candidate_count > 0 and max_cluster_conf >= 0.45
            summary = f"{candidate_count} candidate label region{'s' if candidate_count != 1 else ''} clustered into {cluster_summary.get('region_count', 0)} group{'s' if cluster_summary.get('region_count', 0) != 1 else ''} (confidence {presence_confidence:.2f})"

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
                cluster_method=cluster_summary.get("method", "kmeans"),
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Shipping label analyzer failed: %s", exc, exc_info=True)
            return None
