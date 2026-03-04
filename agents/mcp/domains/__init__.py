"""Domain-specific MCP implementations.

Each domain provides its own tool definitions, registry, interpreter,
and executor — following the same pattern as the core MCP protocol.

Available domains:
- **general** — session management, frame analysis, alerting, evidence, logging
- **pcb** — PCB defect inspection, analytics, reporting, notifications
"""

from .pcb import (
    PCBToolRegistry, PCBInterpreter, PCBExecutor,
    get_pcb_registry, get_pcb_interpreter, get_pcb_executor,
)
from .general import (
    GeneralToolRegistry, MCPInterpreter, MCPExecutor,
    get_agentic_tool_schemas,
)

__all__ = [
    # PCB domain
    "PCBToolRegistry", "PCBInterpreter", "PCBExecutor",
    "get_pcb_registry", "get_pcb_interpreter", "get_pcb_executor",
    # General domain
    "GeneralToolRegistry", "MCPInterpreter", "MCPExecutor",
    "get_agentic_tool_schemas",
]
