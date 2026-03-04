"""Global singleton instances for MCP infrastructure.

Provides thread-safe lazy initialization of the core MCP components.
"""

from __future__ import annotations

import threading
from typing import Optional

from .audit import MCPAuditLog
from .domains.general.executor import MCPExecutor
from .domains.general.interpreter import MCPInterpreter
from .domains.general.tool_defs import GeneralToolRegistry
from .registry import MCPToolRegistry
from .state_machine import AgentStateMachine

_agent_state_machine: Optional[AgentStateMachine] = None
_audit_log: Optional[MCPAuditLog] = None
_tool_registry: Optional[MCPToolRegistry] = None
_mcp_executor: Optional[MCPExecutor] = None
_mcp_interpreter: Optional[MCPInterpreter] = None
_global_lock = threading.RLock()


def get_agent_state_machine() -> AgentStateMachine:
    """Get the global agent state machine instance."""
    global _agent_state_machine
    with _global_lock:
        if _agent_state_machine is None:
            _agent_state_machine = AgentStateMachine()
        return _agent_state_machine


def get_audit_log() -> MCPAuditLog:
    """Get the global audit log instance."""
    global _audit_log
    with _global_lock:
        if _audit_log is None:
            _audit_log = MCPAuditLog()
        return _audit_log


def get_tool_registry() -> MCPToolRegistry:
    """Get the global tool registry instance (general domain)."""
    global _tool_registry
    with _global_lock:
        if _tool_registry is None:
            _tool_registry = GeneralToolRegistry()
        return _tool_registry


def get_mcp_executor() -> MCPExecutor:
    """Get the global MCP executor instance."""
    global _mcp_executor
    with _global_lock:
        if _mcp_executor is None:
            _mcp_executor = MCPExecutor(
                registry=get_tool_registry(),
                state_machine=get_agent_state_machine(),
                audit_log=get_audit_log(),
            )
        return _mcp_executor


def get_mcp_interpreter() -> MCPInterpreter:
    """Get the global MCP interpreter instance."""
    global _mcp_interpreter
    with _global_lock:
        if _mcp_interpreter is None:
            _mcp_interpreter = MCPInterpreter(
                registry=get_tool_registry(),
                audit_log=get_audit_log(),
            )
        return _mcp_interpreter
