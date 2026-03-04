"""General-domain MCP tool definitions — backward-compatible re-export.

The canonical implementation now lives in ``agents.mcp.domains.general.tool_defs``.
This module re-exports all public symbols so existing imports continue to work.
"""

from .domains.general.tool_defs import (  # noqa: F401 — re-export
    AGENTIC_TOOL_DEFS,
    GeneralToolRegistry,
    TOOL_ACKNOWLEDGE_ERROR,
    TOOL_ANALYZE_FRAME,
    TOOL_END_SESSION,
    TOOL_GET_AGENT_STATUS,
    TOOL_GET_SESSION_SUMMARY,
    TOOL_GO_IDLE,
    TOOL_LOG_EVENT,
    TOOL_QUERY_HISTORY,
    TOOL_SAVE_EVIDENCE,
    TOOL_SEND_ALERT_EMAIL,
    TOOL_SET_DETECTION_TASK,
    TOOL_SHUTDOWN_AGENT,
    TOOL_START_MONITORING_SESSION,
    get_agentic_tool_schemas,
)
