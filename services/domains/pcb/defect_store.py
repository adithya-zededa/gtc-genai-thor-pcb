"""Centralized defect query store for monitoring and chat.

Provides advanced query capabilities over the pcb_defects table:
- Time-range queries
- Defect frequency / trend analysis
- Severity ranking
- Source-based aggregation
- Threshold checking

Both the continuous monitoring engine and the interactive chat interface
share this store for consistent, up-to-date defect information.
"""

from __future__ import annotations

import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger

logger = get_logger(__name__)

_store_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


# SQLite CURRENT_TIMESTAMP stores UTC in 'YYYY-MM-DD HH:MM:SS' format.
# We must match both the timezone (UTC) and format (space separator, no
# fractional seconds) for correct string comparison in queries.
_SQLITE_TS_FMT = "%Y-%m-%d %H:%M:%S"


def _utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (matches SQLite)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _time_window(hours: Optional[float] = None) -> Tuple[str, str]:
    """Return (start, end) for the past *hours* in UTC.

    Timestamps are formatted to match SQLite's ``CURRENT_TIMESTAMP``
    (``YYYY-MM-DD HH:MM:SS``, UTC, no fractional seconds).
    """
    end = _utcnow()
    if hours:
        start = end - timedelta(hours=hours)
    else:
        start = datetime(2000, 1, 1)
    return start.strftime(_SQLITE_TS_FMT), end.strftime(_SQLITE_TS_FMT)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_defects_in_range(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    severity: Optional[str] = None,
    defect_type: Optional[str] = None,
    board_type: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Retrieve defects within a time range with optional filters.

    Args:
        start_time: ISO timestamp lower bound (inclusive).
        end_time: ISO timestamp upper bound (inclusive).
        severity: Filter by severity level.
        defect_type: Filter by defect type.
        board_type: Filter by board type.
        limit: Maximum records to return.

    Returns:
        List of defect dicts ordered newest-first.
    """
    from app.database.connection import get_db_connection

    conditions: List[str] = []
    params: List[Any] = []

    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time)
    if severity:
        conditions.append("severity = ?")
        params.append(severity.lower())
    if defect_type:
        conditions.append("LOWER(defect_type) = ?")
        params.append(defect_type.lower())
    if board_type:
        conditions.append("LOWER(board_type) = ?")
        params.append(board_type.lower())

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"SELECT * FROM pcb_defects{where} ORDER BY timestamp DESC LIMIT ?"
    params.append(min(limit, 500))

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    result = [dict(row) for row in rows]
    logger.debug(
        "get_defects_in_range: %d row(s) returned (start=%s, end=%s, "
        "severity=%s, defect_type=%s, board_type=%s, limit=%d)",
        len(result), start_time, end_time, severity, defect_type, board_type, limit,
    )
    return result


def count_defects(
    hours: Optional[float] = None,
    severity: Optional[str] = None,
) -> int:
    """Count defects, optionally within a time window and severity filter."""
    from app.database.connection import get_db_connection

    conditions: List[str] = []
    params: List[Any] = []

    if hours:
        start, _ = _time_window(hours)
        conditions.append("timestamp >= ?")
        params.append(start)
    if severity:
        conditions.append("severity = ?")
        params.append(severity.lower())

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_db_connection() as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM pcb_defects{where}", params).fetchone()

    count = int(row[0]) if row else 0
    logger.debug(
        "count_defects: %d (hours=%s, severity=%s)",
        count, hours, severity,
    )
    return count


def get_latest_defect() -> Optional[Dict[str, Any]]:
    """Return the most recently recorded defect, or None."""
    from app.database.connection import get_db_connection

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM pcb_defects ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

    return dict(row) if row else None


def get_most_severe_defect(
    hours: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Return the highest-severity defect, breaking ties by recency.

    Severity order: high > medium > low.
    """
    from app.database.connection import get_db_connection

    conditions: List[str] = []
    params: List[Any] = []

    if hours:
        start, _ = _time_window(hours)
        conditions.append("timestamp >= ?")
        params.append(start)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_db_connection() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM pcb_defects{where}
            ORDER BY
                CASE severity
                    WHEN 'high' THEN 3
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 1
                    ELSE 0
                END DESC,
                confidence DESC,
                timestamp DESC
            LIMIT 1
            """,
            params,
        ).fetchone()

    return dict(row) if row else None


def get_defect_type_breakdown(
    hours: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Return defect types with their frequencies, ordered by count desc.

    Returns list of {defect_type, count, percentage}.
    """
    from app.database.connection import get_db_connection

    conditions: List[str] = []
    params: List[Any] = []

    if hours:
        start, _ = _time_window(hours)
        conditions.append("timestamp >= ?")
        params.append(start)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT defect_type, COUNT(*) as cnt
            FROM pcb_defects{where}
            GROUP BY defect_type
            ORDER BY cnt DESC
            """,
            params,
        ).fetchall()
        total_row = conn.execute(
            f"SELECT COUNT(*) FROM pcb_defects{where}", params
        ).fetchone()

    total = int(total_row[0]) if total_row else 0
    result = []
    for row in rows:
        cnt = int(row["cnt"])
        result.append({
            "defect_type": row["defect_type"],
            "count": cnt,
            "percentage": round((cnt / total * 100) if total > 0 else 0, 1),
        })
    return result


def get_defect_trend(
    window_hours: float = 168.0,
    bucket_count: int = 7,
) -> Dict[str, Any]:
    """Analyze defect rate trend over the specified window.

    Splits the window into *bucket_count* equal intervals and compares
    the defect rate in the first half vs. the second half.

    Returns:
        {
            "trend": "increasing" | "decreasing" | "stable",
            "buckets": [{period_start, period_end, count}, ...],
            "first_half_rate": float,
            "second_half_rate": float,
            "total_defects": int,
            "window_hours": float,
        }
    """
    from app.database.connection import get_db_connection

    end = _utcnow()
    start = end - timedelta(hours=window_hours)
    bucket_delta = timedelta(hours=window_hours / bucket_count)

    buckets = []
    for i in range(bucket_count):
        b_start = start + bucket_delta * i
        b_end = start + bucket_delta * (i + 1)
        buckets.append({
            "period_start": b_start.strftime(_SQLITE_TS_FMT),
            "period_end": b_end.strftime(_SQLITE_TS_FMT),
            "count": 0,
        })

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT timestamp FROM pcb_defects WHERE timestamp >= ? ORDER BY timestamp ASC",
            (start.strftime(_SQLITE_TS_FMT),),
        ).fetchall()

    total = 0
    for row in rows:
        ts = _parse_ts(row["timestamp"])
        if ts is None:
            continue
        total += 1
        for bucket in buckets:
            b_start = _parse_ts(bucket["period_start"])
            b_end = _parse_ts(bucket["period_end"])
            if b_start is not None and b_end is not None and b_start <= ts < b_end:
                bucket["count"] += 1
                break

    # Compare halves
    mid = bucket_count // 2
    first_half = sum(b["count"] for b in buckets[:mid]) if mid > 0 else 0
    second_half = sum(b["count"] for b in buckets[mid:]) if mid > 0 else 0

    first_rate = first_half / mid if mid > 0 else 0
    second_rate = second_half / (bucket_count - mid) if (bucket_count - mid) > 0 else 0

    # Determine trend with tolerance
    tolerance = 0.15
    if second_rate > first_rate * (1 + tolerance) and first_rate > 0:
        trend = "increasing"
    elif first_rate > second_rate * (1 + tolerance) and second_rate >= 0:
        trend = "decreasing"
    else:
        trend = "stable"

    logger.info(
        "Defect trend analysis: %s over %.0fh (total=%d, "
        "first_half_rate=%.2f, second_half_rate=%.2f)",
        trend, window_hours, total, first_rate, second_rate,
    )

    return {
        "trend": trend,
        "buckets": buckets,
        "first_half_rate": round(first_rate, 2),
        "second_half_rate": round(second_rate, 2),
        "total_defects": total,
        "window_hours": window_hours,
    }


def get_top_defect_sources(
    hours: Optional[float] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return board types / sources producing the most defects.

    Returns list of {board_type, count, severity_breakdown}.
    """
    from app.database.connection import get_db_connection

    conditions: List[str] = []
    params: List[Any] = []

    if hours:
        start, _ = _time_window(hours)
        conditions.append("timestamp >= ?")
        params.append(start)

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT board_type, severity, COUNT(*) as cnt
            FROM pcb_defects{where}
            GROUP BY board_type, severity
            ORDER BY cnt DESC
            """,
            params,
        ).fetchall()

    # Aggregate by board_type
    sources: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bt = row["board_type"]
        if bt not in sources:
            sources[bt] = {"board_type": bt, "count": 0, "severity_breakdown": {}}
        sources[bt]["count"] += int(row["cnt"])
        sources[bt]["severity_breakdown"][row["severity"]] = int(row["cnt"])

    sorted_sources = sorted(sources.values(), key=lambda x: x["count"], reverse=True)
    return sorted_sources[:limit]


def check_threshold_alerts(
    rate_threshold: float = 10.0,
    window_hours: float = 24.0,
    high_severity_threshold: int = 5,
) -> Dict[str, Any]:
    """Check whether defect counts exceed configured thresholds.

    Args:
        rate_threshold: Max acceptable defects per window before alert.
        window_hours: Time window in hours to evaluate.
        high_severity_threshold: Max high-severity defects before alert.

    Returns:
        {
            "threshold_exceeded": bool,
            "alerts": [{"type": ..., "message": ..., "value": ...}],
            "total_defects": int,
            "high_severity_count": int,
            "window_hours": float,
        }
    """
    start, _ = _time_window(window_hours)

    total = count_defects(hours=window_hours)
    high_count = count_defects(hours=window_hours, severity="high")

    alerts: List[Dict[str, Any]] = []

    if total >= rate_threshold:
        alerts.append({
            "type": "rate_exceeded",
            "message": f"Defect count ({total}) exceeds threshold ({rate_threshold}) in the last {window_hours}h",
            "value": total,
            "threshold": rate_threshold,
        })

    if high_count >= high_severity_threshold:
        alerts.append({
            "type": "high_severity_exceeded",
            "message": f"High-severity defects ({high_count}) exceed threshold ({high_severity_threshold}) in the last {window_hours}h",
            "value": high_count,
            "threshold": high_severity_threshold,
        })

    if alerts:
        logger.warning(
            "Threshold alerts triggered: %d alert(s) — total=%d, high=%d, window=%.1fh",
            len(alerts), total, high_count, window_hours,
        )
    else:
        logger.debug(
            "Threshold check OK: total=%d, high=%d, window=%.1fh",
            total, high_count, window_hours,
        )

    return {
        "threshold_exceeded": len(alerts) > 0,
        "alerts": alerts,
        "total_defects": total,
        "high_severity_count": high_count,
        "window_hours": window_hours,
    }


def generate_summary_report(
    period: str = "daily",
    board_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive summary report of monitoring results.

    Args:
        period: "daily" (24h), "weekly" (168h), or "all".
        board_type: Optional board type filter.

    Returns:
        Full report dict with stats, breakdowns, trend, and recommendations.
    """
    hours_map = {"daily": 24.0, "weekly": 168.0, "all": None}
    hours = hours_map.get(period, 24.0)

    defects = get_defects_in_range(
        start_time=_time_window(hours)[0] if hours else None,
        end_time=_utcnow().strftime(_SQLITE_TS_FMT),
        board_type=board_type,
    )
    total = len(defects)
    type_breakdown = get_defect_type_breakdown(hours=hours)
    top_sources = get_top_defect_sources(hours=hours)
    trend_data = get_defect_trend(
        window_hours=hours or 168.0,
        bucket_count=7 if (hours or 168.0) >= 168.0 else max(3, int((hours or 24.0) / 4)),
    )
    threshold_check = check_threshold_alerts(window_hours=hours or 24.0)
    most_severe = get_most_severe_defect(hours=hours)

    # Severity breakdown
    severity_counts = Counter(d.get("severity", "unknown") for d in defects)

    return {
        "success": True,
        "period": period,
        "window_hours": hours,
        "generated_at": _utcnow().strftime(_SQLITE_TS_FMT),
        "filter_board_type": board_type,
        "total_defects": total,
        "severity_breakdown": dict(severity_counts),
        "defect_type_breakdown": type_breakdown,
        "top_sources": top_sources,
        "trend": trend_data,
        "threshold_check": threshold_check,
        "most_severe_defect": most_severe,
        "recent_defects": [d for d in defects[:10]],
    }


def get_defect_insights(
    hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate recommendations and insights based on observed defect patterns.

    Provides actionable analysis:
    - Most common defect types and their likely causes
    - Severity distribution assessment
    - Trend-based recommendations
    - Board-specific risk assessment
    """
    trend = get_defect_trend(window_hours=hours or 168.0)
    type_breakdown = get_defect_type_breakdown(hours=hours)
    top_sources = get_top_defect_sources(hours=hours)
    threshold = check_threshold_alerts(window_hours=hours or 24.0)
    total = count_defects(hours=hours)

    recommendations: List[str] = []
    risk_level = "low"

    # Trend-based recommendations
    if trend["trend"] == "increasing":
        recommendations.append(
            "Defect rate is INCREASING. Investigate recent process changes, "
            "material batches, or equipment calibration drift."
        )
        risk_level = "medium"

    if trend["trend"] == "stable" and total > 0:
        recommendations.append(
            "Defect rate is stable. Current process controls appear effective, "
            "but continue monitoring for anomalies."
        )

    if trend["trend"] == "decreasing":
        recommendations.append(
            "Defect rate is DECREASING — good trend. Recent corrective actions "
            "appear to be working."
        )

    # Type-based recommendations
    if type_breakdown:
        top_type = type_breakdown[0]
        if top_type["percentage"] > 50:
            recommendations.append(
                f"The dominant defect type is '{top_type['defect_type']}' "
                f"({top_type['percentage']}% of all defects). Focus root-cause "
                f"analysis on this category for maximum impact."
            )

    # Source-based recommendations
    if top_sources and len(top_sources) > 1:
        top_src = top_sources[0]
        high_sev = top_src.get("severity_breakdown", {}).get("high", 0)
        if high_sev > 0:
            recommendations.append(
                f"Board type '{top_src['board_type']}' has {high_sev} high-severity "
                f"defect(s). Consider additional quality gates for this board type."
            )
            risk_level = "high"

    # Threshold recommendations
    if threshold["threshold_exceeded"]:
        for alert in threshold["alerts"]:
            recommendations.append(f"ALERT: {alert['message']}")
        risk_level = "high"

    if not recommendations:
        recommendations.append("No significant defect patterns detected. System operating normally.")

    logger.info(
        "Defect insights: risk=%s, trend=%s, total=%d, "
        "recommendations=%d, threshold_exceeded=%s",
        risk_level, trend["trend"], total,
        len(recommendations), threshold["threshold_exceeded"],
    )

    return {
        "success": True,
        "total_defects_analyzed": total,
        "risk_level": risk_level,
        "trend_direction": trend["trend"],
        "recommendations": recommendations,
        "top_defect_type": type_breakdown[0] if type_breakdown else None,
        "top_source": top_sources[0] if top_sources else None,
        "threshold_exceeded": threshold["threshold_exceeded"],
        "analysis_window_hours": hours,
    }
