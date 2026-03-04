"""PCB analysis tools — inspect, classify, and retrieve frames."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.logging import get_logger
from agents.tools.validation import safe_error as _safe_error
from ._helpers import get_inspection_prompt, _auto_log_defect

logger = get_logger(__name__)


def tool_inspect_pcb(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame for PCB defects."""
    logger.info("tool_inspect_pcb invoked (query=%s)", query and query[:80])

    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.CUSTOM,
            custom_prompt=query or get_inspection_prompt(),
        )
    except Exception as exc:
        logger.error("VLM PCB inspection failed: %s", exc)
        return _safe_error("VLM PCB inspection failed", exc=exc)

    if not event:
        logger.warning("PCB inspection returned no event: %s", service.last_error)
        return {
            "success": False,
            "message": service.last_error or "PCB inspection failed — no frame available",
        }

    # Resolve the saved image path so auto-logged defects can reference it
    inspection_image_path = ""
    if hasattr(event, "image_path") and event.image_path:
        inspection_image_path = event.image_path

    logger.info(
        "PCB inspection result: detected=%s confidence=%.2f should_alert=%s image=%s",
        event.detected, event.confidence, event.should_alert, bool(inspection_image_path),
    )

    # Auto-log defect if detected (no LLM round-trip needed)
    _auto_log_defect(
        detected=event.detected,
        full_response=event.full_response,
        description=event.vision_description or "",
        confidence=event.confidence,
        image_path=inspection_image_path,
    )

    return {
        "success": True,
        "message": "PCB inspection complete",
        "data": {
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "should_alert": event.should_alert,
            "full_response": event.full_response,
            "auto_logged": event.detected,  # Inform LLM defect was auto-logged
            "image_path": inspection_image_path,
        },
    }


def tool_classify_board() -> Dict[str, Any]:
    """Identify the board type visible in the current frame."""
    logger.info("tool_classify_board invoked")

    from services.core.monitoring import get_monitoring_service
    from services.domains.pcb.service import classify_board_from_analysis
    from agents.vlm.task_types import TaskType
    import json

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.CUSTOM,
            custom_prompt="Identify the board type visible in this image. Report board_type, board_markings, and any visible text or logos. Respond with ONLY valid JSON.",
        )
    except Exception as exc:
        return _safe_error("VLM board classification failed", exc=exc)

    if not event:
        return {"success": False, "message": "Board classification failed — no frame"}

    analysis: Dict[str, Any] = {}
    if event.full_response and isinstance(event.full_response, str):
        try:
            analysis = json.loads(event.full_response)
            if not isinstance(analysis, dict):
                analysis = {}
        except (json.JSONDecodeError, TypeError):
            analysis = {}

    board_type = classify_board_from_analysis(analysis)

    return {
        "success": True,
        "message": f"Board identified as: {board_type}",
        "data": {
            "board_type": board_type,
            "board_markings": analysis.get("board_markings", ""),
            "confidence": event.confidence,
        },
    }


def tool_get_latest_pcb_frames(
    limit: int = 5,
    unconsumed_only: bool = True,
) -> Dict[str, Any]:
    """Retrieve metadata for the most recently stored PCB frames.

    These frames were automatically captured by the monitoring pipeline
    whenever a PCB was detected on-camera with low motion (< threshold).
    The LLM agent uses this tool to see what frames are available before
    deciding whether to inspect them.
    """
    logger.info(
        "tool_get_latest_pcb_frames invoked (limit=%d, unconsumed_only=%s)",
        limit, unconsumed_only,
    )

    from app.database import PCBFrameStoreRepository

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "message": "limit must be an integer"}
    limit = max(1, min(50, limit))

    try:
        if unconsumed_only:
            frames = PCBFrameStoreRepository.get_latest_unconsumed(limit=limit)
        else:
            frames = PCBFrameStoreRepository.get_latest(limit=limit)
    except Exception as exc:
        return _safe_error("Failed to retrieve stored PCB frames", exc=exc)

    if not frames:
        return {
            "success": True,
            "message": "No stored PCB frames available. The camera may not have detected a PCB yet, or monitoring may not be active.",
            "data": {"frames": [], "count": 0},
        }

    frame_list = [f.to_dict() for f in frames]
    return {
        "success": True,
        "message": f"Found {len(frame_list)} stored PCB frame(s).",
        "data": {"frames": frame_list, "count": len(frame_list)},
    }


def tool_inspect_pcb_frame(
    frame_id: Optional[int] = None,
    query: Optional[str] = None,
    auto_log: bool = False,
) -> Dict[str, Any]:
    """Send a stored PCB frame to the VLM for defect inspection.

    If *frame_id* is omitted the most recent unconsumed frame is used.
    The VLM analysis result is returned to the LLM so it can reason about
    the findings and decide on next steps (e.g., send an alert, log a
    defect, or do nothing).

    By default this tool only analyses.  If ``auto_log=True``, detected
    defects are automatically persisted (and may trigger notifications
    based on notification preferences).
    """
    logger.info(
        "tool_inspect_pcb_frame invoked (frame_id=%s, query=%s)",
        frame_id, query and query[:80],
    )

    import cv2
    from app.database import PCBFrameStoreRepository
    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    # ── Resolve frame ──────────────────────────────────────────────────
    frame_record = None
    if frame_id is not None:
        try:
            frame_record = PCBFrameStoreRepository.get_by_id(int(frame_id))
        except Exception as exc:
            return _safe_error("Failed to look up frame", exc=exc)
    else:
        try:
            unconsumed = PCBFrameStoreRepository.get_latest_unconsumed(limit=1)
            frame_record = unconsumed[0] if unconsumed else None
        except Exception as exc:
            return _safe_error("Failed to retrieve latest frame", exc=exc)

    if frame_record is None:
        return {
            "success": False,
            "message": "No stored PCB frame available to inspect. Wait for the camera to detect a PCB with low motion.",
        }

    # ── Load image from disk ───────────────────────────────────────────
    try:
        frame_img = cv2.imread(frame_record.image_path)
        if frame_img is None:
            return {"success": False, "message": f"Could not read stored frame image: {frame_record.image_path}"}
    except Exception as exc:
        return _safe_error("Failed to load stored frame", exc=exc)

    # ── Run VLM analysis on the stored frame ───────────────────────────
    service = get_monitoring_service()
    if not service or not service.agent:
        return {"success": False, "message": "Monitoring service / agent not available"}

    try:
        event = service.agent.analyze_with_prompt(
            frame=frame_img,
            task_type=TaskType.CUSTOM,
            custom_prompt=query or get_inspection_prompt(),
        )
    except Exception as exc:
        return _safe_error("VLM analysis of stored frame failed", exc=exc)

    if not event:
        return {"success": False, "message": "VLM returned no result for stored frame"}

    # Mark frame as consumed so it doesn't get re-inspected
    try:
        if frame_record.id is not None:
            PCBFrameStoreRepository.mark_consumed(frame_record.id)
    except Exception as exc:
        logger.warning("Failed to mark frame as consumed (frame_id=%s): %s", frame_record.id, exc)

    # Optional side-effect: persist detected defects
    if auto_log:
        _auto_log_defect(
            detected=event.detected,
            full_response=event.full_response,
            description=event.vision_description or "",
            confidence=event.confidence,
            image_path=frame_record.image_path or "",
        )

    return {
        "success": True,
        "message": "Stored PCB frame inspected by VLM",
        "data": {
            "frame_id": frame_record.id,
            "frame_timestamp": frame_record.timestamp,
            "board_signature": frame_record.board_signature,
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "should_alert": event.should_alert,
            "full_response": event.full_response,
            "image_path": frame_record.image_path,
            "auto_logged": bool(auto_log and event.detected),
        },
    }
