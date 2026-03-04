"""Query-history tool implementation."""

from __future__ import annotations

from typing import Any, Dict

from core.logging import get_logger

logger = get_logger(__name__)


def tool_query_history(
    limit: int = 10,
    detected_only: bool = False,
    include_defects: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Query recent detection history with defect information for context.

    Retrieves detection logs from the ``detection_logs`` table and, when
    ``include_defects`` is True (the default), enriches the response with
    recent entries from the ``pcb_defects`` table so the caller gets a
    unified view of what has been detected and logged.
    """
    try:
        from app.database import DetectionLogRepository

        logs, total = DetectionLogRepository.get_paginated(
            page=1,
            per_page=limit,
            detected_only=detected_only,
        )
        log_entries = [log.to_dict() for log in logs]

        result: Dict[str, Any] = {
            "success": True,
            "count": len(log_entries),
            "total": total,
            "events": log_entries,
        }

        # Enrich with defect data from pcb_defects table
        if include_defects:
            try:
                from services.domains.pcb.defect_store import (
                    count_defects,
                    get_defects_in_range,
                    get_latest_defect,
                )

                recent_defects = get_defects_in_range(limit=limit)
                total_defects = count_defects()
                latest_defect = get_latest_defect()
                defects_24h = count_defects(hours=24)

                result["defects"] = {
                    "recent": recent_defects,
                    "total_defects": total_defects,
                    "defects_last_24h": defects_24h,
                    "latest_defect": latest_defect,
                }
                logger.info(
                    "query_history: %d detection log(s), %d defect(s) total, %d in last 24h",
                    len(log_entries), total_defects, defects_24h,
                )
            except Exception as defect_exc:
                logger.warning("Failed to enrich history with defect data: %s", defect_exc)
                result["defects"] = {"error": str(defect_exc)}

        return result
    except Exception as e:
        logger.error("Failed to query history: %s", e)
        return {"success": False, "error": str(e)}
