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
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict

from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import torch
from datetime import datetime
from pathlib import Path
import requests
import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image

import queue
from abc import ABC, abstractmethod
from enum import Enum, auto

# Import from agent_runtime modules
from agent_runtime.config import AgentSettings
from agent_runtime.detection import (
    PackagingBoxAnalysis,
    PackagingBoxAnalyzer,
    RFDetrPackageDetector,
    ShippingLabelAnalysis,
    ShippingLabelAnalyzer,
    DEFAULT_PACKAGING_ANALYZER_PARAMS,
    DEFAULT_RFDETR_CONFIG,
    DEFAULT_SHIPPING_ANALYZER_PARAMS,
)
from agent_runtime.inference import DecisionLLM, OllamaVisionClient
from agent_runtime.models import DetectionDecision
from agent_runtime.state import AgentMemory, DetectionEvent
from agent_runtime.utils import clamp_float, coerce_bool as _coerce_bool
from agent_runtime.alerting import AlertManager



try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional dependency
    structural_similarity = None

try:
    from rfdetr.detr import RFDETRMedium
except ImportError:  # pragma: no cover - optional dependency
    RFDETRMedium = None

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

# Decision LLM configuration
DEFAULT_DECISION_LLM_CONFIG: Dict[str, Any] = {
    "system_prompt": (
        "<|im_start|>system\n"
        "You are a shipping box safety monitor. Your job: detect ANY unlabeled shipping boxes.\n\n"
        "DEFINITIONS:\n"
        "- Shipping box = Brown/tan corrugated cardboard boxes used for delivery/shipping\n"
        "- Shipping label = WHITE or LIGHT-COLORED PAPER STICKER with printed address, barcode, tracking info\n"
        "- NOT shipping boxes = Tissue boxes, cereal boxes, product packaging\n"
        "- NOT shipping labels = Product branding stickers, handwritten text, logos printed on cardboard\n\n"
        "CRITICAL DECISION RULES:\n\n"
        "1. Call trigger_packaging_alert IF:\n"
        "   - At least ONE shipping box is visible, AND\n"
        "   - AT LEAST ONE of those boxes LACKS a proper shipping label\n"
        "   → Even if some boxes have labels, if ANY box is unlabeled, trigger alert!\n\n"
        "2. Call record_no_detection IF:\n"
        "   - No shipping boxes present at all, OR\n"
        "   - ALL shipping boxes have proper shipping labels (not just product stickers), OR\n"
        "   - Scene is too unclear to make a determination\n\n"
        "EXAMPLES:\n"
        "- Vision says '2 boxes, 1 has label' → trigger_packaging_alert (1 box unlabeled)\n"
        "- Vision says '3 boxes, all have labels' → record_no_detection (all labeled)\n"
        "- Vision says '1 box, no label' → trigger_packaging_alert (unlabeled box)\n"
        "- Vision says '1 box with small product sticker, 1 labeled box' → trigger_packaging_alert (first box unlabeled)\n"
    ),
    "user_prompt_template": (
        "<|im_start|>user\n"
        "Vision AI description:\n{vision_description}\n"
        "{extra_context}\n"
        "Analyze and respond with the appropriate tool call.<|im_end|>\n<|im_start|>assistant\n"
    ),
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "trigger_packaging_alert",
                "description": (
                    "ONLY use when a SHIPPING/PACKAGING box (cardboard delivery box) IS ACTUALLY PRESENT in the image "
                    "AND no shipping label is clearly visible on it. "
                    "DO NOT use if: tissue boxes, product boxes, no shipping boxes detected, vision sees only walls/furniture/blur, "
                    "or detectors report 0 boxes. This triggers a critical alert."
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
                    "Use when no alert is needed. This includes: "
                    "(1) NO shipping/packaging boxes detected (only tissue boxes, product boxes, or non-shipping items), "
                    "(2) detectors report 0 boxes, "
                    "(3) shipping boxes present but have visible shipping labels, "
                    "(4) scene is unclear/inconclusive. "
                    "This is the DEFAULT choice when shipping boxes are absent."
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
    ],
}

# --- Architecture Components ---

class ObjectDetector(ABC):
    """Abstract base class for object detection strategies."""
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        pass

class RFDetrAdapter(ObjectDetector):
    """Adapter for the RF-DETR package detector."""
    def __init__(self, config: Dict[str, Any]):
        self.detector = RFDetrPackageDetector(**config)

    def analyze(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        return self.detector.analyze(frame)

class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class CircuitBreaker:
    """Circuit breaker pattern for resilience."""
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 120.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF-OPEN state")
                else:
                    raise RuntimeError("Circuit is OPEN")
            
            # In half-open, we allow one request. If it fails, go back to open.

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state != CircuitState.CLOSED:
                    logger.info("Circuit breaker recovering to CLOSED state")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker tripped to OPEN state. Error: {e}")
            raise

@dataclass
class FrameTask:
    """Data packet for the analysis queue."""
    frame: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionTask:
    """Data packet for the action queue."""
    event: DetectionEvent
    frame: Optional[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

class MonitorDetectionAgent:
    """Main agent class that orchestrates camera monitoring and alert processing."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        detector: ObjectDetector,
        llm_client: OllamaVisionClient,
        circuit_breaker: CircuitBreaker
    ):
        self.config = config
        self.detector = detector
        self.llm_client = llm_client
        self.circuit_breaker = circuit_breaker
        self.config_path = Path(DEFAULT_CONFIG_PATH).expanduser() # Keep for compatibility if needed
        self.last_error: Optional[str] = None
        
        # Initialize components
        ollama_cfg = self.config.get("ollama", {})
        ollama_url = str(ollama_cfg.get("url", "")).rstrip("/")
        decision_model = str(ollama_cfg.get("decision_model", ""))

        decision_llm_cfg = self.config.get("decision_llm") if isinstance(self.config, dict) else {}
        if not isinstance(decision_llm_cfg, dict):
            decision_llm_cfg = {}

        decision_system_prompt = decision_llm_cfg.get(
            "system_prompt",
            DEFAULT_DECISION_LLM_CONFIG["system_prompt"]
        )
        decision_user_prompt = decision_llm_cfg.get(
            "user_prompt_template",
            DEFAULT_DECISION_LLM_CONFIG["user_prompt_template"]
        )
        decision_tools = decision_llm_cfg.get("tools", DEFAULT_DECISION_LLM_CONFIG["tools"])
        if not isinstance(decision_tools, list):
            decision_tools = DEFAULT_DECISION_LLM_CONFIG["tools"]
        else:
            decision_tools = copy.deepcopy(decision_tools)

        decision_timeout = decision_llm_cfg.get("timeout", 30)
        try:
            decision_timeout = int(decision_timeout)
        except (TypeError, ValueError):
            decision_timeout = 30

        memory_cfg = self.config.get("memory") if isinstance(self.config, dict) else {}
        if not isinstance(memory_cfg, dict):
            memory_cfg = {}

        self._memory_max_events = self._safe_positive_int(memory_cfg.get("max_events"), 50)
        default_window = min(10, self._memory_max_events) if self._memory_max_events else 10
        self._memory_summary_window = self._safe_positive_int(
            memory_cfg.get("summary_window"),
            default_window or 1
        )
        self._agent_memory = AgentMemory(
            max_events=self._memory_max_events,
            summary_window=self._memory_summary_window
        )

        analysis_cfg = self.config.get("analysis") if isinstance(self.config, dict) else {}
        if not isinstance(analysis_cfg, dict):
            analysis_cfg = {}

        packaging_cfg = analysis_cfg.get("packaging_box_analyzer")
        if not isinstance(packaging_cfg, dict):
            packaging_cfg = {}

        # Get temperature from ollama config (defaults to 0.1 for deterministic output)
        ollama_temperature = float(ollama_cfg.get("temperature", 0.1) if ollama_cfg.get("temperature") is not None else 0.1)

        self.decision_llm = DecisionLLM(
            ollama_url,
            decision_model,
            system_prompt=decision_system_prompt,
            user_prompt_template=decision_user_prompt,
            tools=decision_tools,
            timeout=decision_timeout,
            temperature=ollama_temperature,
        )
        self.alert_manager = AlertManager(self.config)
        self.packaging_box_analyzer = PackagingBoxAnalyzer(**packaging_cfg)
        
        # RF-DETR is now handled by the injected detector
        self._rfdet_config = analysis_cfg.get("rf_detr", {})
        
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
        self._last_vision_description: Optional[str] = None
        self._last_llm_run_time = 0.0
        self._last_cache_reset = time.time()
        
        # Keyframe tracking to prevent SSIM drift
        self._keyframe_reference = None  # Last frame that triggered full analysis
        self._keyframe_timestamp = 0.0  # When keyframe was captured
        self._keyframe_force_interval = 300.0  # Force re-analysis every 5 minutes
        
        self._apply_runtime_config()
        self.clear_similarity_cache("startup")

        logger.info("MonitorDetectionAgent initialized with dependencies")
    
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

    @staticmethod
    def _safe_positive_int(value: Any, default: int) -> int:
        try:
            candidate = int(value)
            return candidate if candidate > 0 else default
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
        
        # Keyframe force interval configuration
        keyframe_interval_source = os.getenv("AGENT_KEYFRAME_FORCE_INTERVAL_SECONDS")
        if keyframe_interval_source is None:
            keyframe_interval_source = advanced_cfg.get("agent_keyframe_force_interval_seconds")
        self._keyframe_force_interval = max(
            0.0,
            self._safe_float(keyframe_interval_source, self._keyframe_force_interval)
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

    def _vision_descriptions_similar(self, desc1: Optional[str], desc2: Optional[str]) -> bool:
        """
        Check if two vision descriptions are semantically similar enough to reuse a cached decision.
        
        This performs a simple keyword-based comparison focusing on critical elements:
        - Presence/absence of shipping boxes
        - Presence/absence of shipping labels
        - Key object descriptions
        
        Returns True if descriptions are similar enough that the cached decision should still be valid.
        """
        if not desc1 or not desc2:
            return False
        
        # Normalize descriptions for comparison
        d1_lower = desc1.lower()
        d2_lower = desc2.lower()
        
        # Critical keywords that indicate different states
        shipping_label_keywords = [
            'shipping label', 'barcode label', 'label', 'barcode',
            'shipping sticker', 'address label', 'mailing label'
        ]
        
        shipping_box_keywords = [
            'cardboard box', 'shipping box', 'packaging box', 'delivery box',
            'corrugated box', 'brown box', 'package'
        ]
        
        non_shipping_keywords = [
            'tissue box', 'tissue', 'cereal box', 'product box', 'product packaging'
        ]
        
        # Check if presence of shipping labels differs
        has_label_1 = any(keyword in d1_lower for keyword in shipping_label_keywords)
        has_label_2 = any(keyword in d2_lower for keyword in shipping_label_keywords)
        
        if has_label_1 != has_label_2:
            logger.debug("Vision descriptions differ: shipping label presence changed (%s -> %s)", 
                        has_label_1, has_label_2)
            return False
        
        # Check if presence of shipping boxes differs
        has_shipping_box_1 = any(keyword in d1_lower for keyword in shipping_box_keywords)
        has_shipping_box_2 = any(keyword in d2_lower for keyword in shipping_box_keywords)
        
        if has_shipping_box_1 != has_shipping_box_2:
            logger.debug("Vision descriptions differ: shipping box presence changed (%s -> %s)", 
                        has_shipping_box_1, has_shipping_box_2)
            return False
        
        # Check if non-shipping items are mentioned (tissue box, etc.)
        has_non_shipping_1 = any(keyword in d1_lower for keyword in non_shipping_keywords)
        has_non_shipping_2 = any(keyword in d2_lower for keyword in non_shipping_keywords)
        
        if has_non_shipping_1 != has_non_shipping_2:
            logger.debug("Vision descriptions differ: non-shipping item presence changed (%s -> %s)", 
                        has_non_shipping_1, has_non_shipping_2)
            return False
        
        # If all critical elements match, descriptions are similar enough
        return True

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
        analysis_timestamp: float,
        vision_description: Optional[str] = None
    ) -> None:
        """Cache the last evaluated event for SSIM-based reuse."""
        self._last_processed_event = copy.deepcopy(event)
        self._last_vision_description = vision_description
        if reference_frame is not None:
            self._last_similarity_frame = reference_frame
        self._last_llm_run_time = analysis_timestamp
        self._last_cache_reset = analysis_timestamp
        self._remember_event(event, source="analysis")

    def _remember_event(self, event: DetectionEvent, source: str = "analysis") -> None:
        """Persist an event summary in the agent memory ring buffer."""
        try:
            confidence = float(event.confidence) if event.confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        record: Dict[str, Any] = {
            "timestamp": event.timestamp,
            "detected": bool(event.detected),
            "primary_label": event.primary_label or "unknown",
            "confidence": round(confidence, 3),
            "shipping_label_present": event.shipping_label_present
            if isinstance(event.shipping_label_present, bool)
            else None,
            "should_alert": bool(event.should_alert),
            "source": source,
            "tools_used": list(event.tools_used) if isinstance(event.tools_used, list) else [],
        }

        if isinstance(event.decision_trace, dict):
            classification = event.decision_trace.get("classification")
            if classification:
                record["classification"] = classification

            if event.decision_trace.get("agent_similarity_skip"):
                record["source"] = "agent_ssim_guard"

            reasoning = event.decision_trace.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                record["reasoning"] = reasoning.strip()

            scene_metrics = event.decision_trace.get("scene_metrics")
            if isinstance(scene_metrics, dict):
                metrics_payload: Dict[str, Any] = {}
                for key in ("box_count", "label_count", "labels_per_box"):
                    value = scene_metrics.get(key)
                    if value is None:
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(numeric) or math.isinf(numeric):
                        continue
                    if key.endswith("_count"):
                        metrics_payload[key] = int(round(numeric))
                    else:
                        metrics_payload[key] = round(numeric, 3)
                if metrics_payload:
                    record["scene_metrics"] = metrics_payload

        if isinstance(event.vision_description, str) and event.vision_description.strip():
            record["vision_description"] = event.vision_description.strip()

        self._agent_memory.append(record)

    def get_memory_snapshot(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Expose a serialisable view of recent events."""
        return asdict(self._agent_memory.snapshot(limit))

    def summarise_recent_events(self, limit: Optional[int] = None) -> str:
        """Return a short natural language summary of recent agent activity."""
        return self._agent_memory.summarise(limit)

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

        memory_cfg = self.config.get("memory") if isinstance(self.config, dict) else {}
        if not isinstance(memory_cfg, dict):
            memory_cfg = {}
        self._memory_max_events = self._safe_positive_int(
            memory_cfg.get("max_events"),
            getattr(self, "_memory_max_events", 50)
        )
        default_window = min(10, self._memory_max_events) if self._memory_max_events else 10
        self._memory_summary_window = self._safe_positive_int(
            memory_cfg.get("summary_window"),
            default_window or 1
        )
        if hasattr(self, "_agent_memory"):
            self._agent_memory.resize(self._memory_max_events, self._memory_summary_window)
        else:
            self._agent_memory = AgentMemory(
                max_events=self._memory_max_events,
                summary_window=self._memory_summary_window
            )

        analysis_cfg = self.config.get("analysis") if isinstance(self.config, dict) else {}
        if not isinstance(analysis_cfg, dict):
            analysis_cfg = {}

        packaging_cfg = analysis_cfg.get("packaging_box_analyzer")
        if not isinstance(packaging_cfg, dict):
            packaging_cfg = {}
        self.packaging_box_analyzer = PackagingBoxAnalyzer(**packaging_cfg)

        rfdet_cfg = analysis_cfg.get("rf_detr")
        if not isinstance(rfdet_cfg, dict):
            rfdet_cfg = {}
        self._rfdet_config = rfdet_cfg
        self._rfdet_detector = None
        self._rfdet_detector_failed = False

        self._apply_runtime_config()
        self.clear_similarity_cache("config_refresh")

    def clear_similarity_cache(self, reason: str = "manual_reset") -> None:
        """Drop cached SSIM reference data so future frames force fresh analysis."""
        self._last_processed_event = None
        self._last_similarity_frame = None
        self._last_vision_description = None
        self._last_llm_run_time = 0.0
        self._last_cache_reset = time.time()
        
        # Also clear keyframe tracking
        self._keyframe_reference = None
        self._keyframe_timestamp = 0.0

        log_message = f"SSIM cache cleared ({reason})"
        if reason in {"startup", "config_refresh"}:
            logger.info("♻️ %s", log_message)
        elif reason == "ttl_expired":
            logger.debug(log_message)
        else:
            logger.debug(log_message)

    def _check_ssim_cache(self, frame: np.ndarray, analysis_timestamp: float) -> Tuple[bool, Optional[DetectionEvent], Optional[float]]:
        """Check if frame is similar enough to skip analysis."""
        # Check SSIM cache TTL
        if (
            self.agent_ssim_cache_ttl_seconds > 0.0
            and (analysis_timestamp - self._last_cache_reset) >= self.agent_ssim_cache_ttl_seconds
        ):
            logger.info(
                "♻️ SSIM cache TTL %.1fs elapsed; clearing cached decision",
                self.agent_ssim_cache_ttl_seconds
            )
            self.clear_similarity_cache("ttl_expired")

        reference_frame = self._make_similarity_reference(frame)
        similarity_value = None
        force_reanalysis = False
        
        # Check if we need to force re-analysis due to time elapsed since keyframe
        time_since_keyframe = analysis_timestamp - self._keyframe_timestamp
        if time_since_keyframe >= self._keyframe_force_interval:
            logger.info(
                "⏰ Keyframe age %.1fs ≥ %.1fs; forcing re-analysis to prevent drift",
                time_since_keyframe,
                self._keyframe_force_interval
            )
            force_reanalysis = True
            self._keyframe_reference = None  # Clear to establish new keyframe
        
        # Compare against KEYFRAME (last full analysis) instead of last frame
        if (
            reference_frame is not None 
            and self._keyframe_reference is not None 
            and structural_similarity is not None
            and not force_reanalysis
        ):
            try:
                sim_result = structural_similarity(
                    self._keyframe_reference,
                    reference_frame,
                    data_range=255
                )
                similarity_value = float(sim_result[0]) if isinstance(sim_result, tuple) else float(sim_result)
                logger.debug(
                    "Keyframe SSIM: %.3f (threshold: %.3f, age: %.1fs)",
                    similarity_value,
                    self.agent_ssim_threshold,
                    time_since_keyframe
                )
                
                # If similar enough, check if we can reuse the last event
                if similarity_value >= self.agent_ssim_threshold:
                     # Check if we should recheck based on time
                    time_since_last = analysis_timestamp - self._last_llm_run_time
                    if (
                        self._last_processed_event
                        and (
                            self.agent_ssim_recheck_seconds <= 0.0
                            or time_since_last <= self.agent_ssim_recheck_seconds
                        )
                    ):
                        # Reuse event
                        reused_event = copy.deepcopy(self._last_processed_event)
                        if reused_event:
                            reused_event.timestamp = datetime.now().isoformat()
                            reused_event.should_alert = False
                            self._remember_event(reused_event, source="agent_ssim_guard")
                            return True, reused_event, similarity_value

            except Exception as exc:
                logger.debug("Keyframe SSIM comparison failed: %s", exc)
        
        return False, None, similarity_value

    def _perform_computer_vision(self, frame: np.ndarray) -> Tuple[Optional[PackagingBoxAnalysis], Optional[Dict[str, Any]], int, Optional[str], Optional[str]]:
        """Run classical CV and RF-DETR."""
        # Validate frame before processing
        if frame is None:
            logger.debug("Frame is None, skipping CV analysis")
            return None, None, 0, None, None
        
        if not isinstance(frame, np.ndarray):
            logger.debug("Frame is not a numpy array (type: %s), skipping CV analysis", type(frame).__name__)
            return None, None, 0, None, None
        
        if frame.size == 0 or len(frame.shape) < 2:
            logger.debug("Frame has invalid shape: %s, skipping CV analysis", frame.shape if hasattr(frame, 'shape') else 'N/A')
            return None, None, 0, None, None
        
        # Ensure frame is in correct format (BGR with 3 channels)
        if len(frame.shape) == 2:
            # Grayscale frame, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA frame, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.debug("Frame has unexpected shape: %s, skipping CV analysis", frame.shape)
            return None, None, 0, None, None
        
        # Prepare color spaces
        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception as exc:
            logger.warning("Failed to compute color spaces: %s", exc)
            return None, None, 0, None, None

        packaging_analysis = None
        packaging_hint = None
        packaging_box_count = 0
        
        # Stage 1: Classical CV
        if self.packaging_box_analyzer:
            packaging_analysis = self.packaging_box_analyzer.analyze(frame, gray_frame, hsv_frame)
            if packaging_analysis:
                packaging_box_count = max(0, int(packaging_analysis.estimated_box_count))
                packaging_hint = packaging_analysis.summary

        # Stage 2: RF-DETR
        rfdet_analysis = None
        rfdet_hint = None
        if self.detector:
            rfdet_analysis = self.detector.analyze(frame)
            if rfdet_analysis:
                rfdet_box_count = int(rfdet_analysis.get("box_count", 0) or 0)
                if rfdet_box_count > packaging_box_count:
                    packaging_box_count = rfdet_box_count
                rfdet_hint = rfdet_analysis.get("summary")

        return packaging_analysis, rfdet_analysis, packaging_box_count, packaging_hint, rfdet_hint

    def _perform_cognitive_analysis(
        self, 
        frame: np.ndarray, 
        cv_context: Dict[str, Any], 
        similarity_value: Optional[float],
        analysis_timestamp: float
    ) -> Optional[DetectionEvent]:
        """Run Vision LLM and Decision LLM."""
        
        base_prompt = self.config.get("detection", {}).get(
            "prompt",
            "Describe the image in 2-3 concise sentences."
        )
        
        # Circuit Breaker for Vision LLM
        try:
            vision_result = self.circuit_breaker.call(self.llm_client.analyze_frame, frame, base_prompt)
            vision_description = vision_result.get('response', '')
        except RuntimeError as e:
            if "Circuit is OPEN" in str(e):
                logger.warning(
                    "⚡ Vision LLM circuit breaker is OPEN - skipping analysis. "
                    "This happens after multiple timeouts. Will retry in %d seconds.",
                    int(self.circuit_breaker.recovery_timeout)
                )
            else:
                logger.error(f"Vision LLM failed (Circuit Breaker): {e}")
            return None
        except Exception as e:
            logger.error(f"Vision LLM failed: {e}")
            return None

        if not vision_description:
            logger.warning("Vision LLM returned empty response")
            return None

        # Check SSIM cache with vision description comparison
        time_since_last = analysis_timestamp - self._last_llm_run_time
        if (
            self._last_processed_event
            and similarity_value is not None
            and similarity_value >= self.agent_ssim_threshold
            and (
                self.agent_ssim_recheck_seconds <= 0.0
                or time_since_last <= self.agent_ssim_recheck_seconds
            )
        ):
            vision_similar = self._vision_descriptions_similar(vision_description, self._last_vision_description)
            
            if vision_similar:
                logger.info(
                    "♻️ Frame similarity %.3f ≥ threshold %.3f AND vision similar; reusing decision",
                    similarity_value,
                    self.agent_ssim_threshold
                )
                reused_event = copy.deepcopy(self._last_processed_event)
                if reused_event:
                    reused_event.timestamp = datetime.now().isoformat()
                    reused_event.should_alert = False
                    reused_event.vision_description = vision_description
                    self._last_vision_description = vision_description
                    self._remember_event(reused_event, source="agent_ssim_guard")
                    return reused_event

        # Decision LLM
        decision_dict = self.decision_llm.make_detection_decision(
            vision_description,
            cv_context
        )
        
        if decision_dict is None:
            return None
            
        # Convert dict to DetectionDecision
        try:
            decision = DetectionDecision(
                tool=decision_dict.get("tool", "record_no_detection"),
                confidence=float(decision_dict.get("confidence", 0.0)),
                reasoning=decision_dict.get("reasoning", "No reasoning provided"),
                box_count=int(decision_dict.get("box_count", 0)),
                label_count=int(decision_dict.get("label_count", 0)),
                shipping_label_present=decision_dict.get("shipping_label_present"),
                labels_per_box=decision_dict.get("labels_per_box"),
                notes=decision_dict.get("notes")
            )
        except Exception as e:
            logger.error(f"Failed to convert decision dict to model: {e}")
            return None
        
        # Build decision trace
        decision_trace = {
            "classification": "STRUCTURED_DECISION",
            "tool": decision.tool,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "box_count": decision.box_count,
            "label_count": decision.label_count,
            "shipping_label_present": decision.shipping_label_present,
        }
        
        # Determine if alert should be triggered
        should_alert = decision.tool == "trigger_packaging_alert"
        detected = should_alert
        
        # Create detection event
        event = DetectionEvent(
            timestamp=datetime.now().isoformat(),
            detected=detected,
            confidence=decision.confidence,
            primary_label=decision.tool,
            vision_description=vision_description,
            full_response=decision.reasoning,
            should_alert=should_alert,
            shipping_label_present=decision.shipping_label_present,
            tools_used=["vision_llm", "decision_llm"],
            tool_trace=[],
            decision_trace=decision_trace
        )
        return event

    def analyze_frame(self, frame: np.ndarray, metadata: Dict[str, Any]) -> Optional[DetectionEvent]:
        """Orchestrates the analysis pipeline."""
        analysis_started_at = time.time()
        
        # 1. SSIM Check
        should_skip, cached_event, similarity_value = self._check_ssim_cache(frame, analysis_started_at)
        if should_skip and cached_event:
            logger.info("♻️ SSIM skip triggered")
            return cached_event

        # 2. Computer Vision
        packaging_analysis, rfdet_analysis, packaging_box_count, packaging_hint, rfdet_hint = self._perform_computer_vision(frame)
        
        # 3. Fast Fail - No boxes detected
        if packaging_box_count == 0:
            # Track consecutive no-box frames for periodic status updates
            if not hasattr(self, '_no_box_count'):
                self._no_box_count = 0
                self._last_no_box_log_time = 0.0
            
            self._no_box_count += 1
            time_since_last_log = analysis_started_at - self._last_no_box_log_time
            
            # Log every 10 seconds or every 20 frames, whichever comes first
            if time_since_last_log >= 10.0 or self._no_box_count <= 1:
                logger.info(
                    "📦 Scene Status: NO PACKAGING BOXES DETECTED | "
                    "Frames analyzed: %d | Scene is clear - monitoring continues",
                    self._no_box_count
                )
                self._last_no_box_log_time = analysis_started_at
            
            # Update keyframe
            self._keyframe_reference = self._make_similarity_reference(frame)
            self._keyframe_timestamp = analysis_started_at
            
            event = DetectionEvent(
                timestamp=datetime.now().isoformat(),
                detected=False,
                confidence=0.0,
                primary_label="scene_clear",
                vision_description="Scene monitored - no packaging boxes detected",
                full_response=f"Monitoring active: No shipping/packaging boxes in view. Analyzed {self._no_box_count} frames.",
                should_alert=False,
                shipping_label_present=None,
                tools_used=["rf_detr", "classical_cv"],
                tool_trace=[],
                decision_trace={
                    "classification": "SCENE_CLEAR",
                    "reasoning": "RF-DETR object detector found no cardboard boxes in the current frame",
                    "frames_since_last_detection": self._no_box_count
                }
            )
            self._record_last_event(event, self._make_similarity_reference(frame), analysis_started_at)
            return event
        
        # Reset no-box counter when boxes are detected
        if hasattr(self, '_no_box_count'):
            if self._no_box_count > 0:
                logger.info("📦 Box detected after %d clear frames!", self._no_box_count)
            self._no_box_count = 0

        # 4. Cognitive Analysis
        cv_context = {
            "packaging_box_count": packaging_box_count,
            "packaging_hint": packaging_hint,
            "rfdet_hint": rfdet_hint,
        }
        
        if packaging_analysis:
             cv_context["packaging_confidence"] = round(float(packaging_analysis.confidence), 3)
             cv_context["packaging_candidate_count"] = int(packaging_analysis.candidate_count)
        
        if rfdet_analysis:
             cv_context["rfdet_box_count"] = int(rfdet_analysis.get("box_count", 0) or 0)
             avg_conf = rfdet_analysis.get("average_confidence")
             if avg_conf is not None:
                 cv_context["rfdet_average_confidence"] = float(avg_conf)

        event = self._perform_cognitive_analysis(frame, cv_context, similarity_value, analysis_started_at)
        
        if event:
            # Update keyframe
            self._keyframe_reference = self._make_similarity_reference(frame)
            self._keyframe_timestamp = analysis_started_at
            self._record_last_event(event, self._make_similarity_reference(frame), analysis_started_at, event.vision_description)
            
        return event
    

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

# --- Workers ---

class FrameIngestWorker(threading.Thread):
    def __init__(self, device_id: int, queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="IngestWorker")
        self.device_id = device_id
        self.queue = queue
        self.stop_event = stop_event

    def run(self):
        logger.info(f"Starting IngestWorker on device {self.device_id}")
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {self.device_id}")
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue

            try:
                self.queue.put_nowait(FrameTask(frame=frame, timestamp=time.time()))
            except queue.Full:
                # Drop frame
                pass
            
            time.sleep(0.03) # ~30 FPS cap
        
        cap.release()
        logger.info("IngestWorker stopped")

class FrameAnalysisWorker(threading.Thread):
    def __init__(self, agent: MonitorDetectionAgent, input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="AnalysisWorker")
        self.agent = agent
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event

    def run(self):
        logger.info("Starting AnalysisWorker")
        while not self.stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1.0)
                event = self.agent.analyze_frame(task.frame, task.metadata)
                if event:
                    self.output_queue.put(ActionTask(event=event, frame=task.frame, metadata=task.metadata))
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
        logger.info("AnalysisWorker stopped")

class ActionWorker(threading.Thread):
    def __init__(self, agent: MonitorDetectionAgent, queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name="ActionWorker")
        self.agent = agent
        self.queue = queue
        self.stop_event = stop_event

    def run(self):
        logger.info("Starting ActionWorker")
        while not self.stop_event.is_set():
            try:
                task = self.queue.get(timeout=1.0)
                self.agent.process_detection(task.event)
                
                if task.event.detected and self.agent.save_images and task.frame is not None:
                     event_time = datetime.fromisoformat(task.event.timestamp)
                     try:
                         success, buffer = cv2.imencode('.jpg', task.frame)
                         if success:
                             self.agent._save_event_image(
                                buffer.tobytes(),
                                self.agent.images_dir,
                                "detection",
                                event_time,
                                task.metadata
                             )
                     except Exception as e:
                         logger.error(f"Failed to save image in ActionWorker: {e}")

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Action failed: {e}", exc_info=True)
        logger.info("ActionWorker stopped")

class AgentOrchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.stop_event = threading.Event()
        self.analysis_queue = queue.Queue(maxsize=5) # Buffer size
        self.action_queue = queue.Queue(maxsize=50)
        self.threads: List[threading.Thread] = []
        
        # Load config
        self.config = MonitorDetectionAgent.load_config_from_path(config_path)
        
        # Initialize dependencies
        rfdet_cfg = self.config.get("analysis", {}).get("rf_detr", {})
        self.detector = RFDetrAdapter(rfdet_cfg)
        
        ollama_cfg = self.config.get("ollama", {})
        ollama_url = str(ollama_cfg.get("url", "")).rstrip("/")
        vision_model = str(ollama_cfg.get("vision_model", ""))
        ollama_timeout = int(ollama_cfg.get("timeout", 60))
        ollama_temperature = float(ollama_cfg.get("temperature", 0.1) if ollama_cfg.get("temperature") is not None else 0.1)
        
        self.llm_client = OllamaVisionClient(ollama_url, vision_model, timeout=ollama_timeout, temperature=ollama_temperature)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        self.agent = MonitorDetectionAgent(
            self.config,
            self.detector,
            self.llm_client,
            self.circuit_breaker
        )

    def start(self):
        logger.info("Starting Agent Orchestrator...")
        
        # Workers
        ingest = FrameIngestWorker(0, self.analysis_queue, self.stop_event)
        analysis = FrameAnalysisWorker(self.agent, self.analysis_queue, self.action_queue, self.stop_event)
        action = ActionWorker(self.agent, self.action_queue, self.stop_event)
        
        self.threads = [ingest, analysis, action]
        
        for t in self.threads:
            t.start()
            
        logger.info("All workers started")

    def stop(self):
        logger.info("Stopping Agent Orchestrator...")
        self.stop_event.set()
        for t in self.threads:
            t.join()
        logger.info("Agent Orchestrator stopped")

    def run_forever(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    orchestrator.run_forever()
