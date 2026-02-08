"""MCP tool definition and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.logging import get_logger

from .schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema
from .state_machine import AgentState

logger = get_logger(__name__)


@dataclass
class MCPToolDefinition:
    """Complete tool definition with schemas and policies."""
    name: str
    description: str
    category: str
    input_schema: List[MCPParameterSchema]
    output_schema: MCPOutputSchema
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    allowed_in_states: tuple = (AgentState.IDLE, AgentState.MONITORING, AgentState.ANALYZING)
    auto_approve_policy: Optional[Callable[[Dict[str, Any]], bool]] = None
    handler: Optional[Callable[..., Dict[str, Any]]] = None

    def validate_input(self, arguments: Dict[str, Any]) -> tuple:
        """Validate input arguments against schema.

        Returns (is_valid, error_message_or_none).
        """
        for param in self.input_schema:
            value = arguments.get(param.name)

            if param.required and value is None:
                return False, f"Missing required parameter: {param.name}"

            if value is None:
                continue

            if param.type == MCPSchemaType.STRING:
                if not isinstance(value, str):
                    return False, f"Parameter {param.name} must be a string"
                if param.min_length and len(value) < param.min_length:
                    return False, f"Parameter {param.name} must be at least {param.min_length} characters"
                if param.max_length and len(value) > param.max_length:
                    return False, f"Parameter {param.name} must be at most {param.max_length} characters"

            elif param.type == MCPSchemaType.NUMBER:
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be a number"
                if param.min_value is not None and value < param.min_value:
                    return False, f"Parameter {param.name} must be >= {param.min_value}"
                if param.max_value is not None and value > param.max_value:
                    return False, f"Parameter {param.name} must be <= {param.max_value}"

            elif param.type == MCPSchemaType.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    return False, f"Parameter {param.name} must be an integer"

            elif param.type == MCPSchemaType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be a boolean"

            elif param.type == MCPSchemaType.ARRAY:
                if not isinstance(value, list):
                    return False, f"Parameter {param.name} must be an array"

            if param.enum and value not in param.enum:
                return False, f"Parameter {param.name} must be one of: {param.enum}"

        return True, None

    def can_auto_approve(self, arguments: Dict[str, Any]) -> bool:
        """Check if this tool call can be auto-approved by policy."""
        if self.requires_confirmation:
            return False
        if self.auto_approve_policy:
            return self.auto_approve_policy(arguments)
        return not self.requires_confirmation

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to complete JSON Schema format."""
        properties = {}
        required = []

        for param in self.input_schema:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                "returns": self.output_schema.to_json_schema(),
            },
            "metadata": {
                "category": self.category,
                "requires_confirmation": self.requires_confirmation,
                "confirmation_message": self.confirmation_message,
                "allowed_in_states": [s.value for s in self.allowed_in_states],
            },
        }

    def to_display(self) -> Dict[str, Any]:
        """Convert to display format for UI."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_message": self.confirmation_message,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "enum": list(p.enum) if p.enum else None,
                    "default": p.default,
                }
                for p in self.input_schema
            ],
        }


class MCPToolRegistry:
    """Registry of available MCP tools with schema validation."""

    def __init__(self) -> None:
        self._tools: Dict[str, MCPToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools. Override in subclasses."""

    def register(self, tool: MCPToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug("Registered MCP tool: %s", tool.name)

    def get(self, name: str) -> Optional[MCPToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[MCPToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tools_for_state(self, state: AgentState) -> List[MCPToolDefinition]:
        """Get tools available in a given agent state."""
        return [t for t in self._tools.values() if state in t.allowed_in_states]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [tool.to_json_schema() for tool in self._tools.values()]

    def get_display_list(self, state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
        """Get display information for tools, optionally filtered by state."""
        tools = self.get_tools_for_state(state) if state else self.list_tools()
        return [tool.to_display() for tool in tools]
