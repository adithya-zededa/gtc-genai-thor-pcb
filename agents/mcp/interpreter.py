"""General-domain MCP interpreter — backward-compatible re-export.

The canonical implementation now lives in ``agents.mcp.domains.general.interpreter``.
This module re-exports all public symbols so existing imports continue to work.
"""

from .domains.general.interpreter import MCPInterpreter  # noqa: F401 — re-export
