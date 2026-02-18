from agents.classifiers.llm_classifier import ClassificationResult
from agents.mcp.domains.pcb import PCBExecutor, PCBInterpreter, PCBToolRegistry
from agents.mcp.lifecycle import MCPToolCallProposal
from agents.mcp.state_machine import AgentState, AgentStateMachine


class _DummyClassifier:
    def __init__(self, result: ClassificationResult):
        self._result = result

    def classify(self, _message: str) -> ClassificationResult:
        return self._result


def test_pcb_executor_rejects_tool_in_invalid_state():
    state_machine = AgentStateMachine()  # OFF by default
    executor = PCBExecutor(
        registry=PCBToolRegistry(),
        state_machine=state_machine,
        context={},
        invoke_timeout=5,
    )

    proposal = MCPToolCallProposal.create(
        tool_name="inspect_pcb",
        arguments={"query": "inspect for defects"},
        rationale="test",
        confidence=0.9,
        requires_confirmation=False,
    )

    result = executor.submit_proposal(proposal)

    assert result["status"] == "rejected"
    assert "not allowed in state 'off'" in result["reason"]


def test_confirmation_tool_enters_pending_approval_in_valid_state():
    state_machine = AgentStateMachine()
    state_machine.transition_to(
        target=AgentState.IDLE,
        trigger="unit_test",
    )

    executor = PCBExecutor(
        registry=PCBToolRegistry(),
        state_machine=state_machine,
        context={},
        invoke_timeout=5,
    )

    proposal = MCPToolCallProposal.create(
        tool_name="send_defect_alert",
        arguments={
            "recipients": ["qa@example.com"],
            "defect_summary": "solder bridge",
        },
        rationale="test",
        confidence=0.8,
        requires_confirmation=True,
    )

    result = executor.submit_proposal(proposal)

    assert result["status"] == "pending_approval"
    assert result["tool_name"] == "send_defect_alert"


def test_pcb_interpreter_param_allowlist_filters_untrusted_fields(monkeypatch):
    expected = ClassificationResult(
        domain="pcb",
        tool="log_defect",
        confidence=0.95,
        params={
            "board_type": "jetson",
            "defect_type": "bridge",
            "severity": "high",
            "confidence": 0.91,
            "description": "bridge on U1",
            "image_path": "/tmp/a.jpg",
            "shutdown_agent": True,
            "arbitrary": "ignore-me",
        },
        rationale="matched PCB defect logging",
    )

    monkeypatch.setattr(
        "agents.classifiers.llm_classifier.get_classifier",
        lambda: _DummyClassifier(expected),
    )

    interpreter = PCBInterpreter(registry=PCBToolRegistry())
    proposal = interpreter.interpret(
        user_message="log this defect",
        agent_state=AgentState.MONITORING,
        session_id="session_test",
    )

    assert proposal is not None
    assert proposal.tool_name == "log_defect"
    assert proposal.arguments["board_type"] == "jetson"
    assert proposal.arguments["defect_type"] == "bridge"
    assert "shutdown_agent" not in proposal.arguments
    assert "arbitrary" not in proposal.arguments
