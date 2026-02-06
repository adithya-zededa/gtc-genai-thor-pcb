"""PCB domain service."""

from .service import (
    record_defect, should_alert, generate_defect_report,
    classify_board_from_analysis, extract_defects_from_analysis,
)

__all__ = [
    "record_defect", "should_alert", "generate_defect_report",
    "classify_board_from_analysis", "extract_defects_from_analysis",
]
