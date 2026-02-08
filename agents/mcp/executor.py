"""General-domain MCP executor — handles session, analysis, alerts, etc."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

from .audit import AuditEventType, AuditLogEntry, MCPAuditLog
from .executor_base import BaseDomainExecutor
from .lifecycle import MCPToolCallProposal
from .registry import MCPToolRegistry
from .session import MCPSession, SessionType
from .state_machine import AgentState, AgentStateMachine

logger = get_logger(__name__)


class MCPExecutor(BaseDomainExecutor):
    """Executor for the general (non-domain-specific) MCP tools.

    Handles session lifecycle, frame analysis, alerting, evidence,
    logging, history, detection task config, and agent control.
    """

    _domain_label = "general"

    def __init__(
        self,
        registry: MCPToolRegistry,
        state_machine: AgentStateMachine,
        audit_log: MCPAuditLog,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            registry=registry,
            state_machine=state_machine,
            audit_log=audit_log,
            context=context,
        )

    # ── tool dispatch ─────────────────────────────────────────────────────

    def _invoke(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = {
            "start_monitoring_session": self._handle_start_monitoring_session,
            "end_session": self._handle_end_session,
            "get_session_summary": self._handle_get_session_summary,
            "get_agent_status": self._handle_get_agent_status,
            "analyze_current_frame": self._handle_analyze_frame,
            "go_idle": self._handle_go_idle,
            "shutdown_agent": self._handle_shutdown_agent,
            "acknowledge_error": self._handle_acknowledge_error,
            "send_alert_email": self._handle_send_alert_email,
            "save_evidence": self._handle_save_evidence,
            "log_event": self._handle_log_event,
            "query_history": self._handle_query_history,
            "set_detection_task": self._handle_set_detection_task,
        }.get(tool_name)

        if not handler:
            raise ValueError(f"No handler for tool: {tool_name}")
        return handler(arguments)

    # ── handlers ──────────────────────────────────────────────────────────

    def _handle_start_monitoring_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        with self._session_lock:
            if self._current_session and self._current_session.is_active:
                self._current_session.end("Replaced by new session")
                self._sessions.append(self._current_session)

            self._current_session = MCPSession.create(
                SessionType.MONITORING,
                metadata={"description": args.get("description", "")},
            )
            self.state_machine.set_session(self._current_session.id)

        if self.state_machine.state == AgentState.OFF:
            self.state_machine.transition_to(AgentState.IDLE, "start_monitoring_session")

        self.state_machine.transition_to(AgentState.MONITORING, "start_monitoring_session")

        service = get_monitoring_service()
        if service:
            if not service.agent:
                service.initialize()
            service.start_monitoring()

        self.audit_log.log(AuditLogEntry.create(
            event_type=AuditEventType.SESSION_STARTED,
            details={
                "session_id": self._current_session.id,
                "type": self._current_session.type.value,
            },
            session_id=self._current_session.id,
        ))

        return {
            "success": True,
            "message": "Monitoring session started",
            "data": {"session_id": self._current_session.id},
        }

    def _handle_end_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        with self._session_lock:
            if not self._current_session:
                return {"success": False, "message": "No active session"}

            service = get_monitoring_service()
            if service and service.is_monitoring:
                service.stop_monitoring()

            if self.state_machine.state == AgentState.MONITORING:
                self.state_machine.transition_to(AgentState.IDLE, "end_session")

            session_id = self._current_session.id
            self._current_session.end()
            self._sessions.append(self._current_session)

            self.audit_log.log(AuditLogEntry.create(
                event_type=AuditEventType.SESSION_ENDED,
                details={"session_id": session_id},
                session_id=session_id,
            ))

            summary = self._generate_session_summary(self._current_session)
            self._current_session = None
            self.state_machine.set_session(None)

        return {
            "success": True,
            "message": "Session ended",
            "data": {"summary": summary},
        }

    def _handle_get_session_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session_id = args.get("session_id")

        with self._session_lock:
            if session_id:
                session = next((s for s in self._sessions if s.id == session_id), None)
            elif self._current_session:
                session = self._current_session
            elif self._sessions:
                session = self._sessions[-1]
            else:
                return {"success": False, "message": "No sessions available"}

        if not session:
            return {"success": False, "message": f"Session not found: {session_id}"}

        summary = self._generate_session_summary(session)
        return {
            "success": True,
            "message": "Session summary retrieved",
            "data": {"session": session.to_dict(), "summary": summary},
        }

    def _handle_get_agent_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        status = {
            "state": self.state_machine.state.value,
            "state_machine": self.state_machine.get_status(),
            "session": self._current_session.to_dict() if self._current_session else None,
            "monitoring_active": service.is_monitoring if service else False,
            "stats": service._serialize_stats() if service and hasattr(service, '_serialize_stats') else {},
            "metrics": self.audit_log.get_metrics(),
        }
        return {
            "success": True,
            "message": f"Agent is {self.state_machine.state.value}",
            "data": status,
        }

    def _handle_analyze_frame(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        old_state = self.state_machine.state
        if self.state_machine.can_transition_to(AgentState.ANALYZING):
            self.state_machine.transition_to(AgentState.ANALYZING, "analyze_current_frame")

        try:
            service = get_monitoring_service()
            if not service:
                return {"success": False, "message": "Monitoring service not available"}

            query = args.get("query")
            event = service.analyze_single_frame(custom_prompt=query)

            if event:
                return {
                    "success": True,
                    "message": "Frame analyzed",
                    "data": {
                        "detected": event.detected,
                        "confidence": event.confidence,
                        "description": event.vision_description,
                        "should_alert": event.should_alert,
                    },
                }
            return {"success": False, "message": service.last_error or "Analysis failed"}
        finally:
            if self.state_machine.state == AgentState.ANALYZING:
                if self.state_machine.can_transition_to(old_state):
                    self.state_machine.transition_to(old_state, "analysis_complete")
                elif self.state_machine.can_transition_to(AgentState.IDLE):
                    self.state_machine.transition_to(AgentState.IDLE, "analysis_complete")

    def _handle_go_idle(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        service = get_monitoring_service()
        if service and service.is_monitoring:
            service.stop_monitoring()

        self.state_machine.transition_to(AgentState.IDLE, "go_idle")
        return {"success": True, "message": "Agent is now idle"}

    def _handle_shutdown_agent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service

        if self._current_session:
            self._handle_end_session({})

        service = get_monitoring_service()
        if service:
            service.stop_monitoring()

        if self.state_machine.state != AgentState.IDLE:
            if self.state_machine.can_transition_to(AgentState.IDLE):
                self.state_machine.transition_to(AgentState.IDLE, "shutdown_agent")

        self.state_machine.transition_to(AgentState.OFF, "shutdown_agent")
        return {"success": True, "message": "Agent shut down"}

    def _handle_acknowledge_error(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self.state_machine.transition_to(AgentState.IDLE, "acknowledge_error")
        return {"success": True, "message": "Error acknowledged, agent is now idle"}

    def _handle_send_alert_email(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.base import _tool_send_alert_email
        return _tool_send_alert_email(
            recipients=args.get("recipients", []),
            subject=args.get("subject", "Alert"),
            body=args.get("body", ""),
            priority=args.get("priority", "normal"),
            include_image=args.get("include_image", True),
            image_data=self.context.get("image_data"),
        )

    def _handle_save_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.base import _tool_save_evidence
        return _tool_save_evidence(
            image_data=self.context.get("image_data", b""),
            label=args.get("label", "evidence"),
            metadata={"notes": args.get("notes", "")},
        )

    def _handle_log_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.base import _tool_log_event
        return _tool_log_event(
            event_type=args.get("event_type", "general"),
            description=args.get("description", ""),
            severity=args.get("severity", "info"),
        )

    def _handle_query_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from agents.tools.base import _tool_query_history
        return _tool_query_history(
            limit=args.get("limit", 10),
            detected_only=args.get("event_type") == "detection",
        )

    def _handle_set_detection_task(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from services.core.monitoring import get_monitoring_service
        from agents.vlm.task_types import TaskType

        service = get_monitoring_service()
        if not service:
            return {"success": False, "message": "Service not available"}

        task_map = {
            "package_detection": TaskType.PACKAGE_DETECTION,
            "ppe_detection": TaskType.PPE_DETECTION,
            "person_counting": TaskType.PERSON_COUNTING,
            "scene_description": TaskType.SCENE_DESCRIPTION,
            "custom": TaskType.CUSTOM,
        }

        task_type = task_map.get(args.get("task_type"))
        if not task_type:
            return {"success": False, "message": "Invalid task type"}

        service.set_active_prompt(
            task_type=task_type,
            custom_prompt=args.get("custom_instructions", ""),
            alerts_enabled=True,
            agentic_mode=True,
        )

        return {"success": True, "message": f"Detection task set to {args.get('task_type')}"}

    # ── session summary ───────────────────────────────────────────────────

    def _generate_session_summary(self, session: MCPSession) -> str:
        if session.ended_at:
            start = datetime.fromisoformat(session.started_at)
            end = datetime.fromisoformat(session.ended_at)
            minutes = int((end - start).total_seconds() / 60)
            duration = f"{minutes} minutes"
        else:
            start = datetime.fromisoformat(session.started_at)
            minutes = int((datetime.now() - start).total_seconds() / 60)
            duration = f"{minutes} minutes (ongoing)"

        return (
            f"Session '{session.type.value}' ({session.id}):\n"
            f"- Duration: {duration}\n"
            f"- Messages: {session.message_count}\n"
            f"- Tool proposals: {session.tool_proposals}\n"
            f"- Executed: {session.tool_executions}\n"
            f"- Rejected: {session.tool_rejections}"
        )
