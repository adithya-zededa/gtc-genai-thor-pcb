import threading
import time

from agents.core.conveyor_inspection_fsm import (
    ConveyorDecision,
    ConveyorInspectionFSM,
    ConveyorState,
)
from agents.core.state import DetectionEvent


def _make_defect_event() -> DetectionEvent:
    return DetectionEvent(
        timestamp="2026-02-12T00:00:00",
        confidence=0.95,
        primary_label="pcb_defect",
        full_response="defect",
        detected=True,
        should_alert=True,
    )


def test_fsm_starts_in_moving_state():
    decisions = []
    fsm = ConveyorInspectionFSM(
        inspection_window_seconds=0.5,
        on_decision=decisions.append,
    )

    assert fsm.state == ConveyorState.MOVING
    assert fsm.snapshot()["state"] == ConveyorState.MOVING.value
    assert decisions == []


def test_fsm_emits_single_defect_decision_for_board():
    decisions = []
    fsm = ConveyorInspectionFSM(
        inspection_window_seconds=0.5,
        on_decision=decisions.append,
    )

    signature = "board-123"

    fsm.update_inputs(motion_moving=True, board_in_zone=False, board_signature=None)
    fsm.update_inputs(motion_moving=False, board_in_zone=True, board_signature=signature)
    assert fsm.state == ConveyorState.STOPPED_PENDING_INSPECTION

    transitioned = fsm.begin_inspection_on_first_frame()
    assert transitioned is True
    assert fsm.state == ConveyorState.INSPECTING

    fsm.submit_analysis(_make_defect_event())
    assert fsm.state == ConveyorState.INSPECTED_COMPLETE
    assert len(decisions) == 1
    assert decisions[0].decision == ConveyorDecision.DEFECT_FOUND
    assert decisions[0].board_signature == signature

    # Exactly one decision per board.
    fsm.submit_analysis(_make_defect_event())
    assert len(decisions) == 1


def test_fsm_timeout_forces_no_defect_decision():
    decisions = []
    decided = threading.Event()

    def on_decision(payload):
        decisions.append(payload)
        decided.set()

    fsm = ConveyorInspectionFSM(
        inspection_window_seconds=0.05,
        on_decision=on_decision,
    )

    signature = "board-timeout"
    fsm.update_inputs(motion_moving=True, board_in_zone=False, board_signature=None)
    fsm.update_inputs(motion_moving=False, board_in_zone=True, board_signature=signature)
    assert fsm.state == ConveyorState.STOPPED_PENDING_INSPECTION

    assert decided.wait(timeout=1.0)
    assert len(decisions) == 1
    assert decisions[0].decision == ConveyorDecision.NO_DEFECT
    assert decisions[0].board_signature == signature
    assert fsm.state == ConveyorState.INSPECTED_COMPLETE


def test_motion_resume_cancels_active_inspection_and_returns_moving():
    decisions = []
    fsm = ConveyorInspectionFSM(
        inspection_window_seconds=0.5,
        on_decision=decisions.append,
    )

    signature = "board-cancel"
    fsm.update_inputs(motion_moving=True, board_in_zone=False, board_signature=None)
    fsm.update_inputs(motion_moving=False, board_in_zone=True, board_signature=signature)
    assert fsm.begin_inspection_on_first_frame() is True
    assert fsm.state == ConveyorState.INSPECTING

    fsm.update_inputs(motion_moving=True, board_in_zone=True, board_signature=signature)
    assert fsm.state == ConveyorState.MOVING
    assert decisions == []


def test_inspected_board_is_never_reinspected():
    decisions = []
    fsm = ConveyorInspectionFSM(
        inspection_window_seconds=0.5,
        on_decision=decisions.append,
    )

    signature = "board-once"

    fsm.update_inputs(motion_moving=True, board_in_zone=False, board_signature=None)
    fsm.update_inputs(motion_moving=False, board_in_zone=True, board_signature=signature)
    assert fsm.begin_inspection_on_first_frame() is True
    fsm.submit_analysis(_make_defect_event())
    assert len(decisions) == 1

    # Board exits and conveyor moves.
    fsm.update_inputs(motion_moving=True, board_in_zone=False, board_signature=None)
    assert fsm.state == ConveyorState.MOVING

    # Same board stops again in zone: must not re-enter inspection.
    fsm.update_inputs(motion_moving=False, board_in_zone=True, board_signature=signature)
    assert fsm.state == ConveyorState.INSPECTED_COMPLETE
    assert fsm.begin_inspection_on_first_frame() is False
    assert len(decisions) == 1

    # Cleanup to avoid waiting for timer in case of regressions.
    time.sleep(0.01)
