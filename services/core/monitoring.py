"""Camera monitoring service.

Provides the StreamlinedMonitoringService that subscribes to the
camera feed and runs the unified VLM-based detection pipeline.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from agents.core.camera_agent import StreamlinedAgent, CircuitBreaker
from agents.vlm.task_types import TaskType
from agents.core.state import DetectionEvent
from services.core.camera import get_camera_publisher
from services.infrastructure.vlm import create_vlm_client_from_config
from services.infrastructure.config import load_camera_config
from core.logging import get_logger

logger = get_logger(__name__)

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
        self._current_task_type: TaskType = TaskType.PACKAGE_DETECTION
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

    def start_monitoring(self) -> bool:
        """Start the monitoring loop."""
        if self.is_monitoring:
            self.last_error = "Monitoring already active"
            return False
        
        if not self.agent:
            if not self.initialize():
                return False
        
        self.is_monitoring = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitoringService"
        )
        self._thread.start()
        
        logger.info("Monitoring started")
        return True

    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self.is_monitoring = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        logger.info("Monitoring stopped")

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
            return self._run_analysis(
                frame=latest.raw_frame,
                task_type=effective_task,
                custom_prompt=effective_prompt,
                agentic=use_agentic,
            )
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
        if self.agent:
            self.agent.apply_config(config)

    def _run_analysis(
        self,
        frame: Any,
        task_type: TaskType,
        custom_prompt: str,
        agentic: bool,
    ) -> Optional[DetectionEvent]:
        """Run a single analysis pass (agentic or standard)."""
        if agentic:
            config = load_camera_config()
            recipients = (
                config.get("notifications", {})
                .get("email", {})
                .get("recipients", [])
            )
            return self.agent.analyze_agentic(
                frame=frame,
                task_type=task_type,
                custom_prompt=custom_prompt,
                recipients=recipients,
            )
        return self.agent.analyze_with_prompt(
            frame=frame,
            task_type=task_type,
            custom_prompt=custom_prompt,
        )

    def _serialize_stats(self) -> Dict[str, Any]:
        """Get a copy of current stats."""
        with self._stats_lock:
            return self.stats.copy()

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        publisher = self._publisher_getter()
        
        if not publisher.subscribe(self.subscriber_id):
            logger.error("Failed to subscribe to camera feed")
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
            publisher.unsubscribe(self.subscriber_id)

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
                reason=metadata.get("reason"),
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
