"""Model Context Protocol (MCP) — backward-compatible re-export façade.

The implementation has been split into focused modules:
- schema.py        — JSON Schema types (MCPParameterSchema, MCPOutputSchema)
- lifecycle.py     — Proposal / result lifecycle types
- state_machine.py — AgentState enum and AgentStateMachine
- session.py       — SessionType and MCPSession
- audit.py         — AuditEventType, AuditLogEntry, MCPAuditLog
- registry.py      — MCPToolDefinition and MCPToolRegistry
- tool_defs.py     — General-domain tool definitions and GeneralToolRegistry
- interpreter.py   — MCPInterpreter
- executor_base.py — BaseDomainExecutor (shared lifecycle logic)
- executor.py      — MCPExecutor (general domain)
- globals.py       — Singleton accessors

All public symbols are re-exported here so existing
``from agents.mcp.base import …`` statements continue to work.
"""

# Schema
from .schema import MCPSchemaType, MCPParameterSchema, MCPOutputSchema

# Lifecycle
from .lifecycle import ToolLifecycleState, MCPToolCallProposal, MCPToolResult

# State machine
from .state_machine import AgentState, AgentStateMachine

# Session
from .session import SessionType, MCPSession

# Audit
from .audit import AuditEventType, AuditLogEntry, MCPAuditLog

# Registry & tool definitions
from .registry import MCPToolDefinition, MCPToolRegistry

# Interpreter & executor
from .interpreter import MCPInterpreter
from .executor import MCPExecutor

# Global singletons
from .globals import (
    get_agent_state_machine,
    get_audit_log,
    get_tool_registry,
    get_mcp_executor,
    get_mcp_interpreter,
)

__all__ = [
    # Schema
    "MCPSchemaType", "MCPParameterSchema", "MCPOutputSchema",
    # Lifecycle
    "ToolLifecycleState", "MCPToolCallProposal", "MCPToolResult",
    # State machine
    "AgentState", "AgentStateMachine",
    # Session
    "SessionType", "MCPSession",
    # Audit
    "AuditEventType", "AuditLogEntry", "MCPAuditLog",
    # Registry
    "MCPToolDefinition", "MCPToolRegistry",
    # Interpreter & executor
    "MCPInterpreter", "MCPExecutor",
    # Globals
    "get_agent_state_machine", "get_audit_log", "get_tool_registry",
    "get_mcp_executor", "get_mcp_interpreter",
]
