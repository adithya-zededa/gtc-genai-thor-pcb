"""JSON Schema type definitions for MCP tool parameters and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MCPSchemaType(str, Enum):
    """Supported JSON Schema types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass(frozen=True)
class MCPParameterSchema:
    """JSON Schema definition for a tool parameter."""
    name: str
    type: MCPSchemaType
    description: str
    required: bool = False
    enum: Optional[tuple] = None
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    items_type: Optional[MCPSchemaType] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.default is not None:
            schema["default"] = self.default
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.items_type and self.type == MCPSchemaType.ARRAY:
            schema["items"] = {"type": self.items_type.value}
        return schema


@dataclass(frozen=True)
class MCPOutputSchema:
    """JSON Schema definition for tool output."""
    type: MCPSchemaType
    description: str
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_properties: tuple = field(default_factory=tuple)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        if self.properties:
            schema["properties"] = self.properties
        if self.required_properties:
            schema["required"] = list(self.required_properties)
        return schema


def standard_output_schema() -> MCPOutputSchema:
    """Standard output schema used by most tools."""
    return MCPOutputSchema(
        type=MCPSchemaType.OBJECT,
        description="Tool execution result",
        properties={
            "success": {"type": "boolean", "description": "Whether the operation succeeded"},
            "message": {"type": "string", "description": "Human-readable result message"},
            "data": {"type": "object", "description": "Additional result data"},
        },
        required_properties=("success",),
    )
