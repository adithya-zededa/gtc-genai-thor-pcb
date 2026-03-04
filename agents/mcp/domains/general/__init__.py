"""General-domain MCP — session management, frame analysis, alerting, etc.

This package contains the interpreter, executor, and tool definitions
for the general (non-domain-specific) MCP tools.
"""

from .tool_defs import GeneralToolRegistry, get_agentic_tool_schemas, AGENTIC_TOOL_DEFS
from .interpreter import MCPInterpreter
from .executor import MCPExecutor

__all__ = [
    "GeneralToolRegistry",
    "get_agentic_tool_schemas",
    "AGENTIC_TOOL_DEFS",
    "MCPInterpreter",
    "MCPExecutor",
]
