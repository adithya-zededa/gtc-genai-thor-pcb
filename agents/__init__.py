"""Agent modules for AI-powered analysis.

Restructured layout:
    agents/
    ├── core/           - Camera agent, state, alerting
    ├── tools/          - Tool definitions and executors
    ├── mcp/            - MCP base + manager
    │   └── domains/    - PCB, retail domain MCPs
    ├── classifiers/    - LLM intent classifier
    └── vlm/            - Vision Language Model client

All symbols are re-exported here for backward compatibility.
"""

# Core agent components
from .core.camera_agent import StreamlinedAgent, CircuitBreaker
from .core.state import AgentMemory, AgentState, DetectionEvent
from .core.alerting import AlertManager

# Tool system
from .tools.base import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolExecutor,
    TOOL_REGISTRY,
)

# MCP infrastructure
from .mcp.base import (
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
from .mcp.manager import MCPManager, get_mcp_manager

# Classifier
from .classifiers.llm_classifier import LLMIntentClassifier, ClassificationResult, get_classifier

# Domain MCPs
from .mcp.domains.pcb import (
    PCBToolRegistry,
    PCBInterpreter,
    PCBExecutor,
    get_pcb_registry,
    get_pcb_interpreter,
    get_pcb_executor,
)
from .mcp.domains.retail import (
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
    # Classifier
    "LLMIntentClassifier",
    "ClassificationResult",
    "get_classifier",
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
