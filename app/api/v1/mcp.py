"""MCP tools API endpoints with formalized protocol support.

This module provides REST API endpoints for MCP tool operations including:
- Tool listing and schemas
- Proposal submission and approval
- Agent state management
- Session management
- Audit log access

All endpoints follow the MCP principle that tools are PROPOSED, then APPROVED,
then EXECUTED. No tools are executed without explicit approval or policy-based
auto-approval.
"""

from flask import jsonify, request

from agents.mcp.base import (
    AgentState,
    AuditEventType,
    get_agent_state_machine,
    get_audit_log,
    get_mcp_executor,
    get_tool_registry,
)
from agents.mcp.manager import VALID_DOMAINS, get_mcp_manager
from core.logging import get_logger

from . import api_bp

logger = get_logger(__name__)


# =============================================================================
# TOOL ENDPOINTS
# =============================================================================


@api_bp.route("/tools", methods=["GET"])
def list_tools():
    """List all available MCP tools.

    Query params:
        state: Filter tools available in a specific agent state
        domain: Filter by domain (pcb, general). Omit for all.
    """
    state_machine = get_agent_state_machine()
    domain = request.args.get("domain", "").strip() or None

    state_filter = request.args.get("state")
    if state_filter:
        try:
            state = AgentState(state_filter)
        except ValueError:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Invalid state: {state_filter}",
                    }
                ),
                400,
            )
    else:
        state = state_machine.state

    if domain and domain in VALID_DOMAINS:
        # Return tools for a specific domain
        _, _interp, _exec, registry = get_mcp_manager().route("", domain)
        tools = registry.get_display_list(state)
        return jsonify(
            {
                "success": True,
                "domain": domain,
                "tools": tools,
                "count": len(tools),
                "current_state": state.value,
            }
        )

    # Return all tools grouped by domain
    manager = get_mcp_manager()
    all_tools = manager.get_all_tools(state)
    flat_tools = []
    for d, t_list in all_tools.items():
        for t in t_list:
            t["domain"] = d
            flat_tools.append(t)

    return jsonify(
        {
            "success": True,
            "tools": flat_tools,
            "count": len(flat_tools),
            "current_state": state.value,
            "domains": list(all_tools.keys()),
        }
    )


@api_bp.route("/tools/<tool_name>", methods=["GET"])
def get_tool(tool_name: str):
    """Get detailed information about a specific tool."""
    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"Tool not found: {tool_name}",
                }
            ),
            404,
        )

    return jsonify(
        {
            "success": True,
            "tool": tool.to_display(),
            "schema": tool.to_json_schema(),
        }
    )


@api_bp.route("/tools/schemas", methods=["GET"])
def get_tool_schemas():
    """Get JSON schemas for all MCP tools."""
    registry = get_tool_registry()

    return jsonify(
        {
            "success": True,
            "schemas": registry.get_schemas(),
        }
    )


# =============================================================================
# INTERPRETATION ENDPOINTS
# =============================================================================


@api_bp.route("/mcp/interpret", methods=["POST"])
def interpret_message():
    """Interpret a natural language message and produce a tool proposal.

    This is the INTERPRETATION PHASE endpoint.
    No tools are executed - only proposals are generated.

    Request body:
        {
            "message": "Start monitoring the camera",
            "domain": "pcb"  // optional — auto-detected if omitted
        }

    Response:
        {
            "success": true,
            "has_proposal": true,
            "domain": "pcb",
            "proposal": { ... }
        }
    """
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    domain = data.get("domain", request.args.get("domain", "")).strip() or None

    if not message:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No message provided",
                }
            ),
            400,
        )

    manager = get_mcp_manager()
    state_machine = get_agent_state_machine()
    executor = get_mcp_executor()

    session_id = executor.current_session.id if executor.current_session else None

    resolved_domain, proposal = manager.interpret(
        message=message,
        agent_state=state_machine.state,
        session_id=session_id,
        domain=domain,
    )

    if proposal is None:
        return jsonify(
            {
                "success": True,
                "has_proposal": False,
                "domain": resolved_domain,
                "message": "No tool call needed for this message",
            }
        )

    if proposal.is_rejected:
        return jsonify(
            {
                "success": True,
                "has_proposal": True,
                "domain": resolved_domain,
                "proposal": proposal.to_dict(),
                "rejected": True,
                "rejection_reason": proposal.rejection_reason,
            }
        )

    return jsonify(
        {
            "success": True,
            "has_proposal": True,
            "domain": resolved_domain,
            "proposal": proposal.to_dict(),
        }
    )


@api_bp.route("/mcp/submit", methods=["POST"])
def submit_proposal():
    """Submit a tool proposal for approval and execution.

    This submits a proposal to the executor, which will either:
    - Auto-approve and execute (for non-confirmation tools)
    - Queue for user approval (for confirmation-required tools)
    - Reject (for validation or state errors)

    Request body:
        {
            "message": "Start monitoring the camera",
            "domain": "pcb"  // optional — auto-detected if omitted
        }

    Response (auto-executed):
        {
            "success": true,
            "status": "executed",
            "domain": "pcb",
            "result": { ... }
        }

    Response (needs approval):
        {
            "success": true,
            "status": "pending_approval",
            "domain": "pcb",
            "proposal_id": "...",
            "confirmation_message": "..."
        }
    """
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    domain = data.get("domain", request.args.get("domain", "")).strip() or None

    if not message:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No message provided",
                }
            ),
            400,
        )

    manager = get_mcp_manager()
    state_machine = get_agent_state_machine()
    executor = get_mcp_executor()

    session_id = executor.current_session.id if executor.current_session else None

    # Interpret the message through the domain router
    resolved_domain, proposal = manager.interpret(
        message=message,
        agent_state=state_machine.state,
        session_id=session_id,
        domain=domain,
    )

    if proposal is None:
        return jsonify(
            {
                "success": True,
                "status": "no_action",
                "domain": resolved_domain,
                "message": "No tool call needed",
            }
        )

    if proposal.is_rejected:
        return jsonify(
            {
                "success": True,
                "status": "rejected",
                "domain": resolved_domain,
                "reason": proposal.rejection_reason,
                "proposal": proposal.to_dict(),
            }
        )

    # Submit to the correct domain executor
    result = manager.submit(proposal, domain=resolved_domain)

    return jsonify(
        {
            "success": True,
            "domain": resolved_domain,
            **result,
        }
    )


# =============================================================================
# PROPOSAL ENDPOINTS
# =============================================================================


@api_bp.route("/mcp/proposals", methods=["GET"])
def get_pending_proposals():
    """Get all proposals pending approval.

    Query params:
        domain: Filter by domain (pcb, general). Omit for all.
    """
    domain = request.args.get("domain", "").strip() or None
    manager = get_mcp_manager()

    return jsonify(
        {
            "success": True,
            "proposals": manager.get_pending_proposals(domain=domain),
        }
    )


@api_bp.route("/mcp/proposals/<proposal_id>/approve", methods=["POST"])
def approve_proposal(proposal_id: str):
    """Approve a pending proposal for execution.

    This triggers the EXECUTION PHASE for a confirmed tool.
    The manager searches all domain executors for the proposal.
    """
    data = request.get_json() or {}
    domain = data.get("domain", request.args.get("domain", "")).strip() or None

    manager = get_mcp_manager()
    result = manager.approve_proposal(proposal_id, domain=domain)

    return jsonify(
        {
            "success": result["status"] in ("executed", "approved"),
            **result,
        }
    )


@api_bp.route("/mcp/proposals/<proposal_id>/reject", methods=["POST"])
def reject_proposal(proposal_id: str):
    """Reject a pending proposal."""
    data = request.get_json() or {}
    reason = data.get("reason", "Rejected via API")
    domain = data.get("domain", request.args.get("domain", "")).strip() or None

    manager = get_mcp_manager()
    result = manager.reject_proposal(proposal_id, reason, domain=domain)

    return jsonify(
        {
            "success": True,
            **result,
        }
    )


# =============================================================================
# AGENT STATE ENDPOINTS
# =============================================================================


@api_bp.route("/mcp/state", methods=["GET"])
def get_agent_state():
    """Get current agent state and related information."""
    state_machine = get_agent_state_machine()
    executor = get_mcp_executor()
    audit_log = get_audit_log()

    return jsonify(
        {
            "success": True,
            "state": state_machine.state.value,
            "state_machine": state_machine.get_status(),
            "session": (
                executor.current_session.to_dict() if executor.current_session else None
            ),
            "pending_proposals": executor.get_pending_proposals(),
            "metrics": audit_log.get_metrics(),
        }
    )


@api_bp.route("/mcp/state/transitions", methods=["GET"])
def get_state_transitions():
    """Get agent state transition history."""
    state_machine = get_agent_state_machine()
    limit = request.args.get("limit", 50, type=int)

    return jsonify(
        {
            "success": True,
            "transitions": state_machine.get_transitions(limit),
            "current_state": state_machine.state.value,
        }
    )


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================


@api_bp.route("/mcp/session", methods=["GET"])
def get_current_session():
    """Get current MCP session information."""
    executor = get_mcp_executor()
    session = executor.current_session

    if not session:
        return jsonify(
            {
                "success": True,
                "session": None,
                "message": "No active session",
            }
        )

    return jsonify(
        {
            "success": True,
            "session": session.to_dict(),
        }
    )


# =============================================================================
# AUDIT LOG ENDPOINTS
# =============================================================================


@api_bp.route("/mcp/audit", methods=["GET"])
def get_audit_logs():
    """Get audit log entries.

    Query params:
        limit: Maximum entries to return (default 100)
        event_type: Filter by event type
        session_id: Filter by session ID
    """
    audit_log = get_audit_log()
    limit = request.args.get("limit", 100, type=int)
    event_type_str = request.args.get("event_type")
    session_id = request.args.get("session_id")

    # Convert event_type string to enum if provided
    event_type = None
    if event_type_str:
        try:
            event_type = AuditEventType(event_type_str)
        except ValueError:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Invalid event_type: {event_type_str}",
                    }
                ),
                400,
            )

    entries = audit_log.get_entries(
        limit=limit,
        event_type=event_type,
        session_id=session_id,
    )

    return jsonify(
        {
            "success": True,
            "entries": entries,
            "count": len(entries),
        }
    )


@api_bp.route("/mcp/audit/metrics", methods=["GET"])
def get_audit_metrics():
    """Get aggregated audit metrics."""
    audit_log = get_audit_log()

    return jsonify(
        {
            "success": True,
            "metrics": audit_log.get_metrics(),
        }
    )
