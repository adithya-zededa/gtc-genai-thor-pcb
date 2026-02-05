"""Retail billing domain tools for the MCP agent.

Provides tool handler functions used by the Retail MCP executor:
- scan_tray_items: Analyze current frame to identify items on a tray
- lookup_item_price: Look up item prices from the retail catalog DB
- create_bill: Assemble a bill from detected/looked-up items
- generate_invoice: Render an HTML invoice from the current bill
- send_invoice_email: Email the invoice to a recipient
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def tool_scan_tray_items(
    query: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Analyze the current camera frame to identify and count retail items.

    Uses the VLM with ``RETAIL_BILLING`` task type.

    Args:
        query: Optional additional instructions for scanning.

    Returns:
        Analysis result dict with item list and counts.
    """
    from services.monitoring_service import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    custom_prompt = query if query else None
    event = service.analyze_single_frame(
        task_type=TaskType.RETAIL_BILLING,
        custom_prompt=custom_prompt,
    )

    if not event:
        return {
            "success": False,
            "message": service.last_error or "Tray scan failed — no frame available",
        }

    # Try to extract structured items from the VLM response
    items = []
    total_count = 0
    try:
        parsed = json.loads(event.full_response) if isinstance(event.full_response, str) else {}
        items = parsed.get("items", [])
        total_count = parsed.get("total_item_count", len(items))
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "success": True,
        "message": f"Tray scanned: {total_count} item(s) detected",
        "data": {
            "detected": event.detected,
            "confidence": event.confidence,
            "description": event.vision_description,
            "items": items,
            "total_item_count": total_count,
            "full_response": event.full_response,
        },
    }


def tool_lookup_item_price(
    item_name: str = "",
    sku: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Look up an item's price from the retail catalog database.

    Search by name (fuzzy match) or by exact SKU.

    Args:
        item_name: Product name to search for.
        sku: Exact SKU code.

    Returns:
        Dict with matching item(s) and prices.
    """
    from services.retail_service import lookup_items, lookup_item_by_sku

    if sku:
        result = lookup_item_by_sku(sku)
        if result:
            return {
                "success": True,
                "message": f"Found: {result['item_name']} — {result['price']}",
                "data": result,
            }
        return {"success": False, "message": f"No item found with SKU: {sku}"}

    if item_name:
        results = lookup_items([item_name])
        if results and results[0].get("matched"):
            match = results[0]
            return {
                "success": True,
                "message": f"Found: {match['item_name']} — {match['price']}",
                "data": match,
            }
        return {"success": False, "message": f"No catalog match for: {item_name}"}

    return {"success": False, "message": "Provide either item_name or sku to look up"}


def tool_create_bill(
    items: Optional[List[Dict[str, Any]]] = None,
    scan_result: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Assemble a bill from a list of items or a previous scan result.

    If ``items`` is provided directly, it's used as-is.
    If ``scan_result`` is provided (output from scan_tray_items), the detected
    items are matched against the catalog to fill in prices.

    Each item dict should have ``name``, ``quantity``, and optionally ``price``.

    Args:
        items: Explicit list of line items.
        scan_result: Data dict from a previous scan_tray_items call.

    Returns:
        Bill dict with line items, subtotal, tax, total.
    """
    from services.retail_service import calculate_bill, lookup_items

    if not items and scan_result:
        # Build items list from scan result
        scanned_items = scan_result.get("items", [])
        if not scanned_items:
            return {"success": False, "message": "No items found in scan result"}

        # Look up prices from catalog
        names = [it.get("name", "") for it in scanned_items]
        catalog_matches = lookup_items(names)

        items = []
        unmatched = []
        for scanned, matched in zip(scanned_items, catalog_matches):
            entry = {
                "name": matched.get("item_name", scanned.get("name", "Unknown")),
                "quantity": scanned.get("quantity", 1),
                "price": matched.get("price", 0.0),
                "sku": matched.get("sku", ""),
                "category": matched.get("category", scanned.get("category", "")),
            }
            if not matched.get("matched"):
                unmatched.append(scanned.get("name", "Unknown"))
            items.append(entry)

        bill = calculate_bill(items)
        if unmatched:
            bill["warnings"] = [
                f"Price not found in catalog for: {', '.join(unmatched)}. "
                "These items are listed with price 0.00."
            ]
        bill["success"] = True
        bill["message"] = f"Bill created with {bill['unique_items']} item(s), total: {bill['total']}"
        return bill

    if items:
        bill = calculate_bill(items)
        bill["success"] = True
        bill["message"] = f"Bill created with {bill['unique_items']} item(s), total: {bill['total']}"
        return bill

    return {"success": False, "message": "No items provided. Run scan_tray_items first or provide items list."}


def tool_generate_invoice(
    bill: Optional[Dict[str, Any]] = None,
    recipient_email: str = "",
    **kwargs,
) -> Dict[str, Any]:
    """Generate an HTML invoice from a bill and save it to the database.

    Args:
        bill: Bill dict (output from create_bill). If not provided,
              the executor should pass it from context.
        recipient_email: Recipient email for the invoice header.

    Returns:
        Dict with invoice_id and rendered HTML.
    """
    from services.retail_service import generate_invoice_html, save_invoice

    if not bill:
        return {"success": False, "message": "No bill data provided. Run create_bill first."}

    # Save to DB
    save_result = save_invoice(bill, recipient_email)
    if not save_result.get("success"):
        return save_result

    invoice_id = save_result["invoice_id"]

    # Render HTML
    html = generate_invoice_html(bill, recipient_email, invoice_id)

    return {
        "success": True,
        "message": f"Invoice #{invoice_id} generated",
        "data": {
            "invoice_id": invoice_id,
            "html": html,
            "total": bill.get("total", 0),
            "recipient": recipient_email,
        },
    }


def tool_send_invoice_email(
    recipient_email: str = "",
    bill: Optional[Dict[str, Any]] = None,
    invoice_id: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Send an invoice via email.

    If ``bill`` is provided, a new invoice is created, saved, and emailed.
    If ``invoice_id`` is provided, the existing invoice is fetched and emailed.

    Args:
        recipient_email: Email address to send to.
        bill: Bill dict to create and send.
        invoice_id: Existing invoice ID to re-send.

    Returns:
        Dict with success status and email result.
    """
    from services.retail_service import save_and_send_invoice, generate_invoice_html
    from agents.email_tools import send_email

    if not recipient_email:
        return {"success": False, "message": "No recipient email provided"}

    # Path A: full bill provided → save + send
    if bill:
        return save_and_send_invoice(bill, recipient_email)

    # Path B: existing invoice ID
    if invoice_id:
        from app.database.repositories import InvoiceRepository

        invoice = InvoiceRepository.get_by_id(invoice_id)
        if not invoice:
            return {"success": False, "message": f"Invoice #{invoice_id} not found"}

        bill_data = {
            "line_items": invoice.items,
            "subtotal": invoice.subtotal,
            "tax_rate": (invoice.tax / invoice.subtotal) if invoice.subtotal else 0,
            "tax": invoice.tax,
            "total": invoice.total,
            "currency": "INR",
            "generated_at": invoice.timestamp or "",
        }

        html = generate_invoice_html(bill_data, recipient_email, invoice.id)

        try:
            email_result = send_email({
                "to": [recipient_email],
                "subject": f"Invoice #{invoice.id} — INR {invoice.total:.2f}",
                "body": html,
            })
            InvoiceRepository.update_status(invoice.id, "sent")
            return {
                "success": True,
                "message": f"Invoice #{invoice.id} sent to {recipient_email}",
                "email_result": email_result,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "message": "Provide either a bill or an invoice_id to send"}
