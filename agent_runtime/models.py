"""Pydantic models for structured LLM outputs."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DetectionDecision(BaseModel):
    """Structured output from Decision LLM for package detection.
    
    This model enforces schema validation and eliminates fragile JSON parsing.
    """
    tool: Literal["trigger_packaging_alert", "record_no_detection"] = Field(
        ..., 
        description="Action to take based on detection analysis"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score for the detection decision"
    )
    reasoning: str = Field(
        ..., 
        description="Concise explanation of the decision"
    )
    box_count: int = Field(
        default=0,
        ge=0,
        description="Number of boxes detected in the scene"
    )
    label_count: int = Field(
        default=0,
        ge=0,
        description="Number of shipping labels detected"
    )
    shipping_label_present: Optional[bool] = Field(
        default=None,
        description="Whether shipping labels are present on boxes"
    )
    labels_per_box: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average labels per box ratio"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional observations or context"
    )
    
    @field_validator('confidence')
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Ensure confidence is clamped between 0 and 1."""
        return max(0.0, min(1.0, v))
    
    @field_validator('box_count', 'label_count')
    @classmethod
    def ensure_non_negative(cls, v: int) -> int:
        """Ensure counts are non-negative."""
        return max(0, v)
    
    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "tool": "trigger_packaging_alert",
                "confidence": 0.85,
                "reasoning": "Detected cardboard box without visible shipping label",
                "box_count": 1,
                "label_count": 0,
                "shipping_label_present": False,
                "labels_per_box": 0.0,
                "notes": "Box appears to be positioned for shipping"
            }
        }
