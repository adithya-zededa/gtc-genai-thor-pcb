"""Domain-specific services."""

from .pcb import (
    record_defect as pcb_record_defect,
    should_alert as pcb_should_alert,
    generate_defect_report as pcb_generate_defect_report,
    classify_board_from_analysis as pcb_classify_board,
    extract_defects_from_analysis as pcb_extract_defects,
)

__all__ = [
    "pcb_record_defect", "pcb_should_alert", "pcb_generate_defect_report",
    "pcb_classify_board", "pcb_extract_defects",
]
