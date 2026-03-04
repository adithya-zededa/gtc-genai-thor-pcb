"""General-domain MCP executor — backward-compatible re-export.

The canonical implementation now lives in ``agents.mcp.domains.general.executor``.
This module re-exports all public symbols so existing imports continue to work.
"""

from .domains.general.executor import MCPExecutor  # noqa: F401 — re-export
