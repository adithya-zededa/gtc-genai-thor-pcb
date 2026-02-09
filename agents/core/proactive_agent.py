"""Proactive monitoring loop driven entirely by LLM reasoning.

This module implements an intelligent, context-aware camera monitoring agent that:

1. OBSERVES: Continuously watches camera feed and perceives changes
2. REASONS: Maintains temporal context and understands scene state
3. DECIDES: Uses LLM intelligence to determine when/how to analyze frames
4. ADAPTS: Responds to natural-language user instructions dynamically

KEY PRINCIPLES:
- NO hardcoded thresholds (no SSIM, no fixed task switches)
- NO rule-based logic (no "if PCB then analyze")
- ALL decisions come from LLM reasoning over context
- Temporal awareness across frames
- Avoids redundant analysis through scene signature tracking

ARCHITECTURE:
    User Instruction (natural language)
            ↓
    Continuous Monitoring Loop:
        1. Capture frame → 2. LLM Observation (lightweight)
        3. LLM Decision (intelligent) → 4. Execute Action
        5. Update Context → Repeat

The agent maintains rich contextual state and provides it to the LLM at each
decision point, enabling truly intelligent, adaptive behavior.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

from core.logging import get_logger
from agents.vlm.client import UnifiedVLMClient
from agents.vlm.prompts import (
    build_proactive_observation_prompt,
    build_proactive_decision_prompt,
    build_quick_check_prompt,
)
from agents.vlm.task_types import TaskType
from agents.core.state import DetectionEvent
from agents.core.camera_agent import StreamlinedAgent

logger = get_logger(__name__)


class ActionType(str, Enum):
    """Actions that the decision LLM can request."""

    WAIT = "wait"
    QUICK_CHECK = "quick_check"
    FULL_INSPECTION = "full_inspection"

    @classmethod
    def from_value(cls, value: str) -> "ActionType":
        normalized = (value or "wait").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        return cls.WAIT


@dataclass
class ObservationResult:
    """Structured output from the observation step."""

    frame_number: int
    timestamp: str
    scene_summary: str
    primary_objects: List[str]
    target_present: bool
    target_state: str
    scene_changed: bool
    scene_signature: str
    target_ready: bool
    notes: str
    confidence: float
    raw_response: str

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "scene_summary": self.scene_summary,
            "primary_objects": self.primary_objects,
            "target_present": self.target_present,
            "target_state": self.target_state,
            "scene_changed": self.scene_changed,
            "scene_signature": self.scene_signature,
            "target_ready": self.target_ready,
            "notes": self.notes,
            "confidence": round(self.confidence, 3),
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
        }


@dataclass
class DecisionResult:
    """Structured output from the decision step."""

    action: ActionType
    confidence: float
    reasoning: str
    analysis_plan: Dict[str, Any]
    scene_signature: str
    should_emit_event: bool
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


@dataclass
class QuickCheckResult:
    """Structured quick-check confirmation output."""

    target_confirmed: bool
    ready_for_full_inspection: bool
    confidence: float
    notes: str
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_confirmed": self.target_confirmed,
            "ready_for_full_inspection": self.ready_for_full_inspection,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


@dataclass
class MonitoringContext:
    """Rolling context shared with the LLM."""

    instruction: str
    frames_processed: int = 0
    frames_since_scene_change: int = 0
    last_action: ActionType = ActionType.WAIT
    last_action_time: float = 0.0
    last_observation: Optional[ObservationResult] = None
    last_decision: Optional[DecisionResult] = None
    last_quick_check: Optional[QuickCheckResult] = None
    inspected_signatures: Dict[str, float] = field(default_factory=dict)
    quick_check_count: int = 0
    full_inspection_count: int = 0
    target_ready_frames: int = 0

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "frames_processed": self.frames_processed,
            "frames_since_scene_change": self.frames_since_scene_change,
            "last_action": self.last_action.value,
            "seconds_since_last_action": max(0.0, time.time() - self.last_action_time)
            if self.last_action_time else None,
            "inspected_scenes": list(self.inspected_signatures.keys()),
            "quick_checks": self.quick_check_count,
            "full_inspections": self.full_inspection_count,
            "target_ready_frames": self.target_ready_frames,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "frames_processed": self.frames_processed,
            "frames_since_scene_change": self.frames_since_scene_change,
            "last_action": self.last_action.value,
            "last_action_time": self.last_action_time,
            "last_observation": self.last_observation.to_prompt_payload()
            if self.last_observation else None,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
            "last_quick_check": self.last_quick_check.to_dict() if self.last_quick_check else None,
            "inspected_signatures": self.inspected_signatures,
            "quick_check_count": self.quick_check_count,
            "full_inspection_count": self.full_inspection_count,
            "target_ready_frames": self.target_ready_frames,
        }


class ProactiveMonitoringAgent:
    """Intelligent, LLM-driven camera monitoring agent.
    
    This agent continuously monitors a camera feed and uses an LLM to make
    all decisions about when and how to analyze frames. Unlike reactive systems
    with hardcoded rules and thresholds, this agent:
    
    - Understands context from natural language instructions
    - Maintains temporal awareness across frames
    - Reasons about optimal moments for analysis
    - Adapts behavior based on scene changes and history
    - Avoids redundant work through intelligent state tracking
    
    The agent operates in a continuous loop with two LLM stages:
    
    1. OBSERVATION STAGE (cheap/fast):
       - Describes current frame
       - Detects scene changes
       - Identifies target state
       - Updates temporal context
    
    2. DECISION STAGE (intelligent):
       - Decides: wait, quick_check, or full_inspection
       - Based on observation + historical context + user intent
       - Outputs reasoning and confidence
    
    Configuration values are GUIDELINES for the LLM, not rigid rules.
    The LLM makes final decisions based on reasoning, not thresholds.
    """

    DEFAULTS = {
        "frame_interval_seconds": 1.5,  # How often to check frames (guideline)
        "stability_frame_count": 6,  # Hint for scene stability (not a threshold)
        "observation_temperature": 0.1,  # Lower = more consistent observations
        "decision_temperature": 0.2,  # Balanced for intelligent decisions
        "quick_check_temperature": 0.15,  # Quick confirmations
        "max_idle_seconds": 300.0,  # Context hint for decision-making
        "inspection_ttl_seconds": 180.0,  # How long to remember inspected scenes
    }

    def __init__(
        self,
        *,
        instruction: str,
        vlm_client: UnifiedVLMClient,
        detection_agent: StreamlinedAgent,
        publisher_getter: Callable[[], Any],
        event_callback: Optional[Callable[[DetectionEvent, Dict[str, Any]], None]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vlm_client = vlm_client
        self.detection_agent = detection_agent
        self._publisher_getter = publisher_getter
        self._event_callback = event_callback
        self._config = {**self.DEFAULTS, **(config or {})}
        self.context = MonitoringContext(instruction=instruction.strip())

        self.frame_interval = float(self._config.get("frame_interval_seconds", 1.5))
        self.stability_frame_count = int(self._config.get("stability_frame_count", 6))
        self.observation_temperature = float(self._config.get("observation_temperature", 0.1))
        self.decision_temperature = float(self._config.get("decision_temperature", 0.2))
        self.quick_check_temperature = float(self._config.get("quick_check_temperature", 0.15))
        self.max_idle_seconds = float(self._config.get("max_idle_seconds", 300.0))
        self.inspection_ttl = float(self._config.get("inspection_ttl_seconds", 180.0))

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._publisher = None
        self._subscriber_id = f"proactive_{int(time.time()*1000)}"
        self._last_frame_ts = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
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
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._publisher:
            try:
                self._publisher.unsubscribe(self._subscriber_id)
            except Exception:  # pragma: no cover
                pass
        logger.info("Proactive monitoring agent stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def update_instruction(self, instruction: str) -> None:
        self.context.instruction = instruction.strip()
        logger.info("Updated proactive instruction: %s", self.context.instruction)

    def snapshot(self) -> Dict[str, Any]:
        """Get current agent state snapshot for monitoring/debugging."""
        return {
            "config": {
                "frame_interval_seconds": self.frame_interval,
                "stability_frame_count": self.stability_frame_count,
                "observation_temperature": self.observation_temperature,
                "decision_temperature": self.decision_temperature,
                "inspection_ttl_seconds": self.inspection_ttl,
            },
            "context": self.context.snapshot(),
            "running": self._running,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the proactive agent."""
        total_actions = (
            self.context.quick_check_count + 
            self.context.full_inspection_count
        )
        return {
            "frames_processed": self.context.frames_processed,
            "total_actions": total_actions,
            "quick_checks": self.context.quick_check_count,
            "full_inspections": self.context.full_inspection_count,
            "unique_scenes_inspected": len(self.context.inspected_signatures),
            "action_rate": (
                total_actions / self.context.frames_processed 
                if self.context.frames_processed > 0 else 0.0
            ),
            "last_action": self.context.last_action.value,
            "seconds_since_last_action": (
                time.time() - self.context.last_action_time 
                if self.context.last_action_time else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main proactive monitoring loop.
        
        This loop continuously:
        1. Captures frames at configured intervals
        2. Asks LLM to OBSERVE the scene
        3. Asks LLM to DECIDE the next action
        4. Executes the LLM's decision
        5. Updates context for future decisions
        
        All intelligence comes from the LLM - no hardcoded rules.
        """
        self._publisher = self._publisher_getter()
        if not self._publisher.subscribe(self._subscriber_id):
            logger.error("Proactive agent failed to subscribe to camera feed")
            self._running = False
            return

        try:
            while self._running:
                frame_obj = self._publisher.get_frame(self._subscriber_id, timeout=1.0)
                if not frame_obj:
                    continue

                now = time.time()
                if now - self._last_frame_ts < self.frame_interval:
                    continue
                self._last_frame_ts = now

                # ═══════════════════════════════════════════════════════════
                # STAGE 1: LLM OBSERVATION
                # The LLM perceives the scene, tracks changes, identifies targets
                # ═══════════════════════════════════════════════════════════
                observation = self._observe(frame_obj)
                if not observation:
                    continue

                # ═══════════════════════════════════════════════════════════
                # STAGE 2: LLM DECISION
                # The LLM reasons over observation + context to decide action
                # ═══════════════════════════════════════════════════════════
                decision = self._decide(frame_obj, observation)
                self.context.last_observation = observation
                self.context.last_decision = decision
                self.context.frames_processed += 1

                # ═══════════════════════════════════════════════════════════
                # EXECUTE THE LLM'S DECISION
                # No rules here - just execute what the LLM decided
                # ═══════════════════════════════════════════════════════════
                if decision.action == ActionType.WAIT:
                    self._register_action(decision.action)
                    logger.debug(
                        "LLM decided: WAIT | Reasoning: %s",
                        decision.reasoning[:80]
                    )
                    continue

                if decision.action == ActionType.QUICK_CHECK:
                    quick_result = self._run_quick_check(frame_obj, observation)
                    if quick_result:
                        self.context.last_quick_check = quick_result
                        self.context.quick_check_count += 1
                        logger.debug(
                            "LLM decided: QUICK_CHECK | Confirmed=%s Ready=%s",
                            quick_result.target_confirmed,
                            quick_result.ready_for_full_inspection
                        )
                    self._register_action(decision.action)
                    continue

                if decision.action == ActionType.FULL_INSPECTION:
                    logger.info(
                        "LLM decided: FULL_INSPECTION | Reasoning: %s",
                        decision.reasoning[:100]
                    )
                    event = self._run_full_inspection(frame_obj, observation, decision)
                    self._register_action(decision.action)
                    if event and self._event_callback:
                        metadata = {
                            "frame_number": frame_obj.frame_number,
                            "reason": "proactive_full_inspection",
                            "scene_signature": observation.scene_signature,
                            "decision": decision.to_dict(),
                        }
                        self._event_callback(event, metadata)
        except Exception as exc:  # pragma: no cover
            logger.error("Proactive monitoring loop crashed: %s", exc, exc_info=True)
        finally:
            try:
                self._publisher.unsubscribe(self._subscriber_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # LLM interaction helpers
    # ------------------------------------------------------------------

    def _observe(self, frame_obj) -> Optional[ObservationResult]:
        context_hint = {
            "frames_since_scene_change": self.context.frames_since_scene_change,
            "last_action": self.context.last_action.value,
            "frames_processed": self.context.frames_processed,
            "target_ready_frames": self.context.target_ready_frames,
        }
        prompt = build_proactive_observation_prompt(
            self.context.instruction,
            self.context.last_observation.to_prompt_payload() if self.context.last_observation else None,
            context_hint,
        )

        response = self.vlm_client.run_structured_prompt(
            frame_obj.raw_frame,
            prompt,
            temperature=self.observation_temperature,
            max_tokens=400,
        )
        parsed = response.get("parsed") or {}
        if not parsed:
            logger.debug("Observation step returned no JSON; skipping frame")
            return None

        scene_summary = str(parsed.get("scene_summary", "No summary"))
        primary_objects = parsed.get("primary_objects") or []
        if not isinstance(primary_objects, list):
            primary_objects = [str(primary_objects)]
        scene_signature = parsed.get("scene_signature") or self._derive_signature(
            frame_obj.frame_number,
            scene_summary,
        )
        observation = ObservationResult(
            frame_number=frame_obj.frame_number,
            timestamp=frame_obj.timestamp,
            scene_summary=scene_summary,
            primary_objects=[str(o) for o in primary_objects][:6],
            target_present=bool(parsed.get("target_present", False)),
            target_state=str(parsed.get("target_state", "unknown")),
            scene_changed=bool(parsed.get("scene_changed", False)),
            scene_signature=scene_signature,
            target_ready=bool(parsed.get("target_ready", False)),
            notes=str(parsed.get("notes", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            raw_response=response.get("raw", ""),
        )

        if observation.scene_changed:
            self.context.frames_since_scene_change = 0
        else:
            self.context.frames_since_scene_change += 1

        if observation.target_ready:
            self.context.target_ready_frames = min(
                self.context.target_ready_frames + 1,
                self.stability_frame_count * 2,
            )
        else:
            self.context.target_ready_frames = 0

        return observation

    def _decide(self, frame_obj, observation: ObservationResult) -> DecisionResult:
        self._expire_inspections()
        context_payload = self.context.to_prompt_payload()
        context_payload["recent_scene_stability_frames"] = self.context.frames_since_scene_change
        context_payload["stability_frame_goal"] = self.stability_frame_count
        context_payload["max_idle_seconds"] = self.max_idle_seconds
        context_payload["already_inspected"] = observation.scene_signature in self.context.inspected_signatures

        prompt = build_proactive_decision_prompt(
            self.context.instruction,
            observation.to_prompt_payload(),
            context_payload,
        )
        response = self.vlm_client.run_structured_prompt(
            frame_obj.raw_frame,
            prompt,
            temperature=self.decision_temperature,
            max_tokens=512,
        )
        parsed = response.get("parsed") or {}
        action = ActionType.from_value(parsed.get("action"))
        decision = DecisionResult(
            action=action,
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", ""))[:400],
            analysis_plan=parsed.get("analysis_plan") or {},
            scene_signature=str(parsed.get("scene_signature", observation.scene_signature)),
            should_emit_event=bool(parsed.get("should_emit_event", True)),
            raw_response=response.get("raw", ""),
        )
        return decision

    def _run_quick_check(self, frame_obj, observation: ObservationResult) -> Optional[QuickCheckResult]:
        prompt = build_quick_check_prompt(
            self.context.instruction,
            observation.to_prompt_payload(),
            self.context.to_prompt_payload(),
        )
        response = self.vlm_client.run_structured_prompt(
            frame_obj.raw_frame,
            prompt,
            temperature=self.quick_check_temperature,
            max_tokens=256,
        )
        parsed = response.get("parsed") or {}
        if not parsed:
            return None
        return QuickCheckResult(
            target_confirmed=bool(parsed.get("target_confirmed", False)),
            ready_for_full_inspection=bool(parsed.get("ready_for_full_inspection", False)),
            confidence=float(parsed.get("confidence", 0.5)),
            notes=str(parsed.get("notes", ""))[:200],
            raw_response=response.get("raw", ""),
        )

    def _run_full_inspection(
        self,
        frame_obj,
        observation: ObservationResult,
        decision: DecisionResult,
    ) -> Optional[DetectionEvent]:
        task_type, custom_prompt = self._resolve_plan(decision.analysis_plan)
        custom_text = custom_prompt or self.context.instruction
        event = self.detection_agent.analyze_with_prompt(
            frame=frame_obj.raw_frame,
            task_type=task_type,
            custom_prompt=custom_text,
        )
        if event:
            event.decision_trace.setdefault("proactive_context", {})
            event.decision_trace["proactive_context"].update({
                "scene_signature": observation.scene_signature,
                "action": decision.action.value,
                "analysis_plan": decision.analysis_plan,
                "decision_confidence": decision.confidence,
            })
            self.context.full_inspection_count += 1
            self.context.inspected_signatures[observation.scene_signature] = time.time()
        return event

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _register_action(self, action: ActionType) -> None:
        self.context.last_action = action
        self.context.last_action_time = time.time()

    def _expire_inspections(self) -> None:
        cutoff = time.time() - self.inspection_ttl
        stale = [sig for sig, ts in self.context.inspected_signatures.items() if ts < cutoff]
        for sig in stale:
            self.context.inspected_signatures.pop(sig, None)

    @staticmethod
    def _derive_signature(frame_number: int, summary: str) -> str:
        safe_summary = summary.lower().split(".")[0][:24]
        safe_summary = safe_summary.replace(" ", "-")
        return f"scene-{frame_number}-{safe_summary or 'unknown'}"

    @staticmethod
    def _resolve_plan(plan: Dict[str, Any]) -> tuple[TaskType, Optional[str]]:
        task_name = str(plan.get("task", "")).strip().lower()
        mapping = {
            "package_detection": TaskType.PACKAGE_DETECTION,
            "pcb_inspection": TaskType.PCB_INSPECTION,
            "ppe_detection": TaskType.PPE_DETECTION,
            "person_counting": TaskType.PERSON_COUNTING,
            "scene_description": TaskType.SCENE_DESCRIPTION,
            "retail_billing": TaskType.RETAIL_BILLING,
            "custom": TaskType.CUSTOM,
        }
        task_type = mapping.get(task_name, TaskType.CUSTOM)
        custom_prompt = plan.get("custom_prompt")
        if task_type != TaskType.CUSTOM and custom_prompt:
            # Keep custom instructions as an overlay even for known tasks
            return task_type, custom_prompt
        return task_type, custom_prompt
*** End File