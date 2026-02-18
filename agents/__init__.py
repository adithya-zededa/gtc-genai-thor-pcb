"""Agent modules for AI-powered analysis.

Restructured layout::

    agents/
    ├── core/           - Camera agent, state, alerting
    ├── tools/          - Tool definitions and executors
    │   └── validation  - Shared input-validation helpers
    ├── mcp/            - MCP base + domain executors
    │   ├── schema      - Schema / parameter types
    │   ├── lifecycle   - Proposal & result lifecycle
    │   ├── state_machine - Agent operational state
    │   ├── session     - Session management
    │   ├── audit       - Audit logging
    │   ├── registry    - Tool definition & registry
    │   ├── executor_base - Shared executor logic
    │   ├── interpreter - General-domain interpreter
    │   ├── executor    - General-domain executor
    │   ├── globals     - Singleton accessors
    │   ├── manager     - Domain router
    │   └── domains/    - PCB domain MCP
    ├── classifiers/    - LLM intent classifier
    └── vlm/            - Vision Language Model client

Subpackage ``__init__`` files re-export their own public API.
Import directly from subpackages for clarity::

    from agents.core import StreamlinedAgent
    from agents.mcp  import get_mcp_manager
    from agents.classifiers import get_classifier

The symbols below are re-exported *only* for backward compatibility with
code that does ``from agents import X``.  New code should prefer the
qualified subpackage imports shown above.
"""

# ── Core ──────────────────────────────────────────────────────────────────
from .core import (
    StreamlinedAgent,
    CircuitBreaker,
    AgentMemory,
    AgentSnapshot,
    DetectionEvent,
)

# Backward-compat alias kept by core/__init__.py
from .core import AgentState  # noqa: F811  (alias for AgentSnapshot)

# ── MCP (via façade) ─────────────────────────────────────────────────────
from .mcp import MCPManager, get_mcp_manager

# ── Classifier ────────────────────────────────────────────────────────────
from .classifiers import LLMIntentClassifier, ClassificationResult, get_classifier

__all__ = [
    # Core
    "StreamlinedAgent",
    "CircuitBreaker",
    "AgentMemory",
    "AgentSnapshot",
    "AgentState",
    "DetectionEvent",
    # MCP Manager (entry-point for tool calling)
    "MCPManager",
    "get_mcp_manager",
    # Classifier
    "LLMIntentClassifier",
    "ClassificationResult",
    "get_classifier",
]
