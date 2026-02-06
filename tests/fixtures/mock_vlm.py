"""Mock VLM client fixtures for testing without a real inference backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockDetectionResult:
    """Simulated VLM detection result for testing."""
    detected: bool = False
    confidence: float = 0.0
    primary_label: str = "no_detection"
    full_response: str = "No objects detected"
    vision_description: str = ""


class MockVLMClient:
    """Fake VLM client that returns predetermined responses.
    
    Usage::
    
        client = MockVLMClient(default_detected=True, default_confidence=0.95)
        result = client.analyze(frame)
    """

    def __init__(
        self,
        default_detected: bool = False,
        default_confidence: float = 0.0,
        default_label: str = "no_detection",
    ) -> None:
        self.default_detected = default_detected
        self.default_confidence = default_confidence
        self.default_label = default_label
        self.call_count = 0
        self.last_frame = None

    def analyze(self, frame, **kwargs) -> MockDetectionResult:
        self.call_count += 1
        self.last_frame = frame
        return MockDetectionResult(
            detected=self.default_detected,
            confidence=self.default_confidence,
            primary_label=self.default_label,
        )
