"""Conversational memory for multi-purpose vision agents.

This module provides memory capabilities for agents that need to:
1. Track conversation history with the VLM
2. Remember recent analysis results for context
3. Maintain task-specific state across frames
4. Support dynamic prompt changes during monitoring
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry representing an interaction."""
    
    timestamp: str
    task_type: str
    user_prompt: Optional[str]
    detected: bool
    confidence: float
    reasoning: str
    should_alert: bool
    details: Dict[str, Any] = field(default_factory=dict)
    frame_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_type": self.task_type,
            "user_prompt": self.user_prompt,
            "detected": self.detected,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "should_alert": self.should_alert,
            "details": self.details,
            "frame_number": self.frame_number,
        }
    
    def to_context_string(self) -> str:
        """Convert to a concise context string for VLM prompts."""
        parts = [f"[{self.timestamp}]"]
        if self.user_prompt:
            parts.append(f"Task: {self.user_prompt[:50]}")
        parts.append(f"Result: {'Detected' if self.detected else 'Not detected'}")
        if self.detected:
            parts.append(f"(conf: {self.confidence:.0%})")
        if self.reasoning:
            parts.append(f"- {self.reasoning[:100]}")
        return " ".join(parts)


@dataclass  
class ConversationTurn:
    """Represents a single conversation turn with the VLM."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "task_type": self.task_type,
        }


class ConversationalMemory:
    """Thread-safe conversational memory for vision agents.
    
    Maintains:
    1. Recent analysis history (ring buffer)
    2. Conversation context for the VLM
    3. Current active task/prompt
    4. Task-specific counters and statistics
    
    Example usage:
        memory = ConversationalMemory(max_history=50, max_conversation=10)
        
        # Set active task
        memory.set_active_prompt("Count all people wearing red shirts")
        
        # Record an analysis
        memory.record_analysis(
            task_type="custom",
            detected=True,
            confidence=0.85,
            reasoning="Found 3 people in red shirts",
            details={"count": 3},
        )
        
        # Get context for next VLM call
        context = memory.get_context_for_prompt()
    """
    
    def __init__(
        self,
        max_history: int = 50,
        max_conversation: int = 10,
        context_window: int = 5,
    ):
        """Initialize conversational memory.
        
        Args:
            max_history: Maximum number of analysis results to store
            max_conversation: Maximum conversation turns to maintain
            context_window: Number of recent results to include in context
        """
        self._max_history = max(1, int(max_history))
        self._max_conversation = max(1, int(max_conversation))
        self._context_window = max(1, min(context_window, self._max_history))
        
        self._history: Deque[MemoryEntry] = deque(maxlen=self._max_history)
        self._conversation: Deque[ConversationTurn] = deque(maxlen=self._max_conversation)
        
        # Current state
        self._active_prompt: Optional[str] = None
        self._active_task_type: str = "package_detection"
        self._frame_counter: int = 0
        self._session_start: str = datetime.now().isoformat()
        
        # Task-specific counters
        self._task_counts: Dict[str, Dict[str, int]] = {}
        
        self._lock = threading.RLock()
        
        logger.info(
            "ConversationalMemory initialized: max_history=%d, max_conversation=%d",
            self._max_history,
            self._max_conversation,
        )
    
    # =========================================================================
    # Active Prompt Management
    # =========================================================================
    
    def set_active_prompt(
        self,
        prompt: str,
        task_type: str = "custom",
    ) -> None:
        """Set the current active prompt for analysis.
        
        Args:
            prompt: The user's prompt/instruction
            task_type: Type of task (package_detection, ppe_detection, custom, etc.)
        """
        with self._lock:
            self._active_prompt = prompt
            self._active_task_type = task_type
            
            # Record in conversation
            self._conversation.append(ConversationTurn(
                role="user",
                content=prompt,
                task_type=task_type,
            ))
            
            logger.info("Active prompt set: %s (task: %s)", prompt[:50], task_type)
    
    def get_active_prompt(self) -> Tuple[Optional[str], str]:
        """Get the current active prompt and task type.
        
        Returns:
            Tuple of (prompt, task_type)
        """
        with self._lock:
            return self._active_prompt, self._active_task_type
    
    def clear_active_prompt(self) -> None:
        """Clear the active prompt, reverting to default behavior."""
        with self._lock:
            self._active_prompt = None
            self._active_task_type = "package_detection"
            logger.info("Active prompt cleared")
    
    # =========================================================================
    # Analysis Recording
    # =========================================================================
    
    def record_analysis(
        self,
        task_type: str,
        detected: bool,
        confidence: float,
        reasoning: str,
        should_alert: bool = False,
        details: Optional[Dict[str, Any]] = None,
        user_prompt: Optional[str] = None,
    ) -> MemoryEntry:
        """Record an analysis result.
        
        Args:
            task_type: Type of analysis performed
            detected: Whether the target was detected
            confidence: Detection confidence (0-1)
            reasoning: VLM's reasoning/explanation
            should_alert: Whether this triggers an alert
            details: Task-specific details dict
            user_prompt: The prompt used for this analysis
            
        Returns:
            The created MemoryEntry
        """
        with self._lock:
            self._frame_counter += 1
            
            entry = MemoryEntry(
                timestamp=datetime.now().isoformat(),
                task_type=task_type,
                user_prompt=user_prompt or self._active_prompt,
                detected=detected,
                confidence=confidence,
                reasoning=reasoning,
                should_alert=should_alert,
                details=details or {},
                frame_number=self._frame_counter,
            )
            
            self._history.append(entry)
            
            # Update task counters
            if task_type not in self._task_counts:
                self._task_counts[task_type] = {
                    "total": 0,
                    "detections": 0,
                    "alerts": 0,
                }
            self._task_counts[task_type]["total"] += 1
            if detected:
                self._task_counts[task_type]["detections"] += 1
            if should_alert:
                self._task_counts[task_type]["alerts"] += 1
            
            # Record VLM response in conversation
            self._conversation.append(ConversationTurn(
                role="assistant",
                content=reasoning,
                task_type=task_type,
            ))
            
            return entry
    
    # =========================================================================
    # Context Building
    # =========================================================================
    
    def get_context_for_prompt(self, include_history: bool = True) -> str:
        """Build context string to include in VLM prompts.
        
        Args:
            include_history: Whether to include recent analysis history
            
        Returns:
            Context string to prepend/append to prompts
        """
        with self._lock:
            parts = []
            
            # Add session context
            parts.append(f"Session started: {self._session_start}")
            parts.append(f"Frames analyzed: {self._frame_counter}")
            
            # Add task statistics
            if self._task_counts:
                stats_parts = []
                for task_type, counts in self._task_counts.items():
                    stats_parts.append(
                        f"{task_type}: {counts['detections']}/{counts['total']} detections"
                    )
                parts.append("Stats: " + ", ".join(stats_parts))
            
            # Add recent history
            if include_history and self._history:
                recent = list(self._history)[-self._context_window:]
                if recent:
                    parts.append("\nRecent observations:")
                    for entry in recent:
                        parts.append(f"  - {entry.to_context_string()}")
            
            return "\n".join(parts)
    
    def get_conversation_context(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation history in chat format.
        
        Args:
            max_turns: Maximum number of turns to return
            
        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts
        """
        with self._lock:
            turns = list(self._conversation)
            if max_turns:
                turns = turns[-max_turns:]
            return [{"role": t.role, "content": t.content} for t in turns]
    
    # =========================================================================
    # History Access
    # =========================================================================
    
    def get_recent_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent analysis history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of memory entry dicts, newest last
        """
        with self._lock:
            entries = list(self._history)
            if limit:
                entries = entries[-limit:]
            return [e.to_dict() for e in entries]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict with statistics about stored data
        """
        with self._lock:
            history_list = list(self._history)
            
            total = len(history_list)
            detections = sum(1 for e in history_list if e.detected)
            alerts = sum(1 for e in history_list if e.should_alert)
            
            # Calculate average confidence for detections
            detection_confidences = [e.confidence for e in history_list if e.detected]
            avg_confidence = (
                sum(detection_confidences) / len(detection_confidences)
                if detection_confidences else 0.0
            )
            
            return {
                "session_start": self._session_start,
                "frame_counter": self._frame_counter,
                "history_size": total,
                "history_max": self._max_history,
                "conversation_size": len(self._conversation),
                "conversation_max": self._max_conversation,
                "active_prompt": self._active_prompt,
                "active_task_type": self._active_task_type,
                "total_analyses": total,
                "total_detections": detections,
                "total_alerts": alerts,
                "detection_rate": detections / total if total else 0.0,
                "average_detection_confidence": round(avg_confidence, 3),
                "task_counts": dict(self._task_counts),
            }
    
    def summarize(self, limit: Optional[int] = None) -> str:
        """Generate a human-readable summary.
        
        Args:
            limit: Limit history to recent N entries
            
        Returns:
            Summary string
        """
        with self._lock:
            stats = self.get_statistics()
            
            parts = []
            parts.append(f"Session active since {stats['session_start']}")
            parts.append(f"Analyzed {stats['frame_counter']} frames")
            
            if stats['total_analyses'] > 0:
                parts.append(
                    f"Detection rate: {stats['detection_rate']:.1%} "
                    f"({stats['total_detections']}/{stats['total_analyses']})"
                )
                parts.append(f"Alerts triggered: {stats['total_alerts']}")
                
                if stats['average_detection_confidence'] > 0:
                    parts.append(
                        f"Average detection confidence: {stats['average_detection_confidence']:.1%}"
                    )
            
            if self._active_prompt:
                parts.append(f"\nCurrent task: {self._active_prompt}")
            
            return " | ".join(parts)
    
    # =========================================================================
    # Memory Management
    # =========================================================================
    
    def clear(self) -> None:
        """Clear all memory."""
        with self._lock:
            self._history.clear()
            self._conversation.clear()
            self._task_counts.clear()
            self._frame_counter = 0
            self._active_prompt = None
            self._active_task_type = "package_detection"
            self._session_start = datetime.now().isoformat()
            logger.info("Memory cleared")
    
    def resize(
        self,
        max_history: Optional[int] = None,
        max_conversation: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> None:
        """Resize memory buffers.
        
        Args:
            max_history: New max history size
            max_conversation: New max conversation size
            context_window: New context window size
        """
        with self._lock:
            if max_history is not None:
                new_max = max(1, int(max_history))
                preserved = list(self._history)[-new_max:]
                self._history = deque(preserved, maxlen=new_max)
                self._max_history = new_max
            
            if max_conversation is not None:
                new_max = max(1, int(max_conversation))
                preserved = list(self._conversation)[-new_max:]
                self._conversation = deque(preserved, maxlen=new_max)
                self._max_conversation = new_max
            
            if context_window is not None:
                self._context_window = max(1, min(int(context_window), self._max_history))
            
            logger.info(
                "Memory resized: max_history=%d, max_conversation=%d, context_window=%d",
                self._max_history,
                self._max_conversation,
                self._context_window,
            )


# Convenience function for building memory-enhanced prompts
def build_memory_enhanced_prompt(
    base_prompt: str,
    memory: ConversationalMemory,
    include_context: bool = True,
    context_position: str = "after",  # "before" or "after"
) -> str:
    """Build a prompt enhanced with memory context.
    
    Args:
        base_prompt: The base prompt template
        memory: ConversationalMemory instance
        include_context: Whether to include memory context
        context_position: Where to add context ("before" or "after")
        
    Returns:
        Enhanced prompt string
    """
    if not include_context:
        return base_prompt
    
    context = memory.get_context_for_prompt()
    
    if context_position == "before":
        return f"{context}\n\n{base_prompt}"
    else:
        return f"{base_prompt}\n\n---\nContext from previous observations:\n{context}"
