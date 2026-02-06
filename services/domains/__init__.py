"""Domain-specific services."""

from .pcb import (
    record_defect as pcb_record_defect,
    should_alert as pcb_should_alert,
    generate_defect_report as pcb_generate_defect_report,
    classify_board_from_analysis as pcb_classify_board,
    extract_defects_from_analysis as pcb_extract_defects,
)
from .retail import (
    lookup_items as retail_lookup_items,
    lookup_item_by_sku as retail_lookup_item_by_sku,
    calculate_bill as retail_calculate_bill,
    generate_invoice_html as retail_generate_invoice_html,
    save_invoice as retail_save_invoice,
    save_and_send_invoice as retail_save_and_send_invoice,
)

__all__ = [
    "pcb_record_defect", "pcb_should_alert", "pcb_generate_defect_report",
    "pcb_classify_board", "pcb_extract_defects",
    "retail_lookup_items", "retail_lookup_item_by_sku",
    "retail_calculate_bill", "retail_generate_invoice_html",
    "retail_save_invoice", "retail_save_and_send_invoice",
]
