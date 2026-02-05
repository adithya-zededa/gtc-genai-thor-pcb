"""Agent modules for AI-powered analysis."""

from .camera_agent import StreamlinedAgent, CircuitBreaker
from .state import AgentMemory, AgentState, DetectionEvent
from .alerting import AlertManager
from .tools import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolExecutor,
    TOOL_REGISTRY,
)
from .mcp import (
    # Core types
    MCPSchemaType,
    MCPParameterSchema,
    MCPOutputSchema,
    # Lifecycle
    ToolLifecycleState,
    MCPToolCallProposal,
    MCPToolResult,
    # State machine
    AgentState as MCPAgentState,
    AgentStateMachine,
    # Session
    SessionType,
    MCPSession,
    # Tools
    MCPToolDefinition,
    MCPToolRegistry,
    # Audit
    AuditEventType,
    AuditLogEntry,
    MCPAuditLog,
    # Interpreter & Executor
    MCPInterpreter,
    MCPExecutor,
    # Global accessors
    get_agent_state_machine,
    get_audit_log,
    get_tool_registry,
    get_mcp_executor,
    get_mcp_interpreter,
)
from .mcp_manager import MCPManager, get_mcp_manager
from .pcb_mcp import (
    PCBToolRegistry,
    PCBInterpreter,
    PCBExecutor,
    get_pcb_registry,
    get_pcb_interpreter,
    get_pcb_executor,
)
from .retail_mcp import (
    RetailToolRegistry,
    RetailInterpreter,
    RetailExecutor,
    get_retail_registry,
    get_retail_interpreter,
    get_retail_executor,
)

__all__ = [
    # Camera agent
    "StreamlinedAgent",
    "CircuitBreaker",
    # State management (legacy)
    "AgentMemory",
    "AgentState",
    "DetectionEvent",
    # Alerting
    "AlertManager",
    # Tools (legacy)
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "TOOL_REGISTRY",
    # MCP Core
    "MCPSchemaType",
    "MCPParameterSchema",
    "MCPOutputSchema",
    "ToolLifecycleState",
    "MCPToolCallProposal",
    "MCPToolResult",
    "MCPAgentState",
    "AgentStateMachine",
    "SessionType",
    "MCPSession",
    "MCPToolDefinition",
    "MCPToolRegistry",
    "AuditEventType",
    "AuditLogEntry",
    "MCPAuditLog",
    "MCPInterpreter",
    "MCPExecutor",
    "get_agent_state_machine",
    "get_audit_log",
    "get_tool_registry",
    "get_mcp_executor",
    "get_mcp_interpreter",
    # MCP Manager
    "MCPManager",
    "get_mcp_manager",
    # PCB MCP
    "PCBToolRegistry",
    "PCBInterpreter",
    "PCBExecutor",
    "get_pcb_registry",
    "get_pcb_interpreter",
    "get_pcb_executor",
    # Retail MCP
    "RetailToolRegistry",
    "RetailInterpreter",
    "RetailExecutor",
    "get_retail_registry",
    "get_retail_interpreter",
    "get_retail_executor",
]
