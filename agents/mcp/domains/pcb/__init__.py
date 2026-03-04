"""PCB Inspection domain MCP package.

Provides a self-contained MCP for PCB defect detection workflows.
Reuses the core MCP infrastructure via ``BaseDomainExecutor`` and
registers only PCB-specific tools and intent phrases.

Capabilities
~~~~~~~~~~~~
- **PCB analysis**: get/inspect stored frames, live inspection, board classification.
- **Defect management**: log, report, and query defects.
- **Monitoring analytics**: status, trends, severity ranking, source analysis.
- **Notification control**: enable/disable/configure email notifications via chat.
- **Reporting**: daily/weekly/on-demand summary reports with threshold checking.
- **Insights**: AI-driven recommendations based on observed defect patterns.
"""

from __future__ import annotations

import threading
from typing import Optional

from .tool_defs import PCBToolRegistry, TOOL_PARAM_ALLOWLIST, HOURS_AWARE_TOOLS
from .interpreter import PCBInterpreter
from .executor import PCBExecutor

__all__ = [
    "PCBToolRegistry",
    "PCBInterpreter",
    "PCBExecutor",
    "TOOL_PARAM_ALLOWLIST",
    "HOURS_AWARE_TOOLS",
    "get_pcb_registry",
    "get_pcb_interpreter",
    "get_pcb_executor",
]


# ── Singletons ────────────────────────────────────────────────────────────

_pcb_registry: Optional[PCBToolRegistry] = None
_pcb_interpreter: Optional[PCBInterpreter] = None
_pcb_executor: Optional[PCBExecutor] = None
_pcb_lock = threading.RLock()


def get_pcb_registry() -> PCBToolRegistry:
    global _pcb_registry
    with _pcb_lock:
        if _pcb_registry is None:
            _pcb_registry = PCBToolRegistry()
        return _pcb_registry


def get_pcb_interpreter() -> PCBInterpreter:
    global _pcb_interpreter
    with _pcb_lock:
        if _pcb_interpreter is None:
            _pcb_interpreter = PCBInterpreter(get_pcb_registry())
        return _pcb_interpreter


def get_pcb_executor() -> PCBExecutor:
    global _pcb_executor
    with _pcb_lock:
        if _pcb_executor is None:
            _pcb_executor = PCBExecutor(get_pcb_registry())
        return _pcb_executor
