"""Retail billing domain tools for the MCP agent.

Provides thin-adapter tool handler functions used by the Retail MCP executor.
Each function validates its inputs, delegates to the retail_service layer,
and returns a sanitised result dict.

Tools:
- scan_tray_items: Analyze current frame to identify items on a tray
- lookup_item_price: Look up item prices from the retail catalog DB
- create_bill: Assemble a bill from detected/looked-up items
- generate_invoice: Render an HTML invoice from the current bill
- send_invoice_email: Email the invoice to a recipient
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
MAX_EMAIL_LEN = 254
MAX_QUANTITY = 9999
MIN_QUANTITY = 1


def _validate_email(email: str) -> Optional[str]:
    """Return *None* if valid, or an error message."""
    if not email:
        return "Email address is required"
    if len(email) > MAX_EMAIL_LEN:
        return f"Email address too long (max {MAX_EMAIL_LEN} chars)"
    if not _EMAIL_RE.match(email):
        return f"Invalid email format: {email}"
    return None


def _safe_error(internal_msg: str, *, exc: Optional[Exception] = None) -> Dict[str, Any]:
    """Return a user-safe error dict and log the internal detail."""
    if exc:
        logger.error("%s: %s", internal_msg, exc, exc_info=True)
    else:
        logger.error(internal_msg)
    return {"success": False, "message": "An internal error occurred. Please try again."}


def _clamp_quantity(raw: Any) -> int:
    """Coerce *raw* to an int in [MIN_QUANTITY, MAX_QUANTITY]."""
    try:
        q = int(raw)
    except (ValueError, TypeError):
        q = 1
    return max(MIN_QUANTITY, min(MAX_QUANTITY, q))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_scan_tray_items(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame to identify and count retail items.

    Uses the VLM with ``RETAIL_BILLING`` task type.
    """
    logger.info("tool_scan_tray_items invoked (query=%s)", query and query[:80])

    from services.core.monitoring import get_monitoring_service
    from agents.vlm.task_types import TaskType

    service = get_monitoring_service()
    if not service:
        return {"success": False, "message": "Monitoring service not available"}

    try:
        event = service.analyze_single_frame(
            task_type=TaskType.RETAIL_BILLING,
            custom_prompt=query or None,
        )
    except Exception as exc:
        return _safe_error("VLM tray scan failed", exc=exc)

    if not event:
        return {
            "success": False,
            "message": service.last_error or "Tray scan failed — no frame available",
        }

    # Try to extract structured items from the VLM response
    items: List[Dict[str, Any]] = []
    total_count = 0
    if event.full_response and isinstance(event.full_response, str):
        try:
            parsed = json.loads(event.full_response)
            raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
            # Sanitise each item — clamp quantities
            for it in raw_items:
                it["quantity"] = _clamp_quantity(it.get("quantity", 1))
            items = raw_items
            total_count = parsed.get("total_item_count", len(items)) if isinstance(parsed, dict) else len(items)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("VLM response was not valid JSON, returning raw: %s", exc)

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
) -> Dict[str, Any]:
    """Look up an item's price from the retail catalog database."""
    logger.info("tool_lookup_item_price invoked (item_name=%s, sku=%s)", item_name, sku)

    from services.domains.retail.service import lookup_items, lookup_item_by_sku

    try:
        if sku:
            result = lookup_item_by_sku(sku.strip()[:50])
            if result:
                return {
                    "success": True,
                    "message": f"Found: {result['item_name']} — {result['price']}",
                    "data": result,
                }
            return {"success": False, "message": f"No item found with SKU: {sku}"}

        if item_name:
            results = lookup_items([item_name.strip()[:200]])
            if results and results[0].get("matched"):
                match = results[0]
                return {
                    "success": True,
                    "message": f"Found: {match['item_name']} — {match['price']}",
                    "data": match,
                }
            return {"success": False, "message": f"No catalog match for: {item_name}"}

        return {"success": False, "message": "Provide either item_name or sku to look up"}

    except Exception as exc:
        return _safe_error("Item lookup failed", exc=exc)


def tool_create_bill(
    items: Optional[List[Dict[str, Any]]] = None,
    scan_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a bill from a list of items or a previous scan result.

    Each item dict should have ``name``, ``quantity``, and optionally ``price``.
    """
    logger.info(
        "tool_create_bill invoked (explicit_items=%s, has_scan=%s)",
        bool(items), bool(scan_result),
    )

    from services.domains.retail.service import calculate_bill, lookup_items

    try:
        if not items and scan_result:
            scanned_items = scan_result.get("items", [])
            if not scanned_items:
                return {"success": False, "message": "No items found in scan result"}

            names = [it.get("name", "") for it in scanned_items]
            catalog_matches = lookup_items(names)

            items = []
            unmatched = []
            for scanned, matched in zip(scanned_items, catalog_matches):
                entry = {
                    "name": matched.get("item_name", scanned.get("name", "Unknown")),
                    "quantity": _clamp_quantity(scanned.get("quantity", 1)),
                    "price": float(matched.get("price", 0.0)),
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
            # Validate and sanitise
            for it in items:
                it["quantity"] = _clamp_quantity(it.get("quantity", 1))
                it["price"] = max(0.0, float(it.get("price", 0.0)))

            bill = calculate_bill(items)
            bill["success"] = True
            bill["message"] = f"Bill created with {bill['unique_items']} item(s), total: {bill['total']}"
            return bill

        return {
            "success": False,
            "message": "No items provided. Run scan_tray_items first or provide items list.",
        }

    except Exception as exc:
        return _safe_error("Bill creation failed", exc=exc)


def tool_generate_invoice(
    bill: Optional[Dict[str, Any]] = None,
    recipient_email: str = "",
) -> Dict[str, Any]:
    """Generate an HTML invoice from a bill and save it to the database."""
    logger.info("tool_generate_invoice invoked (has_bill=%s, email=%s)", bool(bill), recipient_email)

    from services.domains.retail.service import generate_invoice_html, save_invoice

    if not bill:
        return {"success": False, "message": "No bill data provided. Run create_bill first."}

    # Validate email if provided
    if recipient_email:
        err = _validate_email(recipient_email)
        if err:
            return {"success": False, "message": err}

    try:
        save_result = save_invoice(bill, recipient_email)
        if not save_result.get("success"):
            return {"success": False, "message": "Failed to save invoice"}

        invoice_id = save_result["invoice_id"]
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

    except Exception as exc:
        return _safe_error("Invoice generation failed", exc=exc)


def tool_send_invoice_email(
    recipient_email: str = "",
    bill: Optional[Dict[str, Any]] = None,
    invoice_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Send an invoice via email.

    If ``bill`` is provided, a new invoice is created, saved, and emailed.
    If ``invoice_id`` is provided, the existing invoice is fetched and emailed.
    """
    logger.info(
        "tool_send_invoice_email invoked (email=%s, has_bill=%s, invoice_id=%s)",
        recipient_email, bool(bill), invoice_id,
    )

    # Validate recipient
    err = _validate_email(recipient_email)
    if err:
        return {"success": False, "message": err}

    from services.domains.retail.service import save_and_send_invoice, generate_invoice_html
    from agents.tools.email import send_email

    try:
        # Path A: full bill provided -> save + send
        if bill:
            return save_and_send_invoice(bill, recipient_email)

        # Path B: existing invoice ID
        if invoice_id:
            from app.database.repositories import InvoiceRepository

            invoice = InvoiceRepository.get_by_id(invoice_id)
            if not invoice:
                return {"success": False, "message": f"Invoice #{invoice_id} not found"}

            currency = os.getenv("RETAIL_CURRENCY", "INR")
            bill_data = {
                "line_items": invoice.items,
                "subtotal": invoice.subtotal,
                "tax_rate": (invoice.tax / invoice.subtotal) if invoice.subtotal else 0,
                "tax": invoice.tax,
                "total": invoice.total,
                "currency": currency,
                "generated_at": invoice.timestamp or "",
            }

            html = generate_invoice_html(bill_data, recipient_email, invoice.id)

            email_result = send_email({
                "to": [recipient_email],
                "subject": f"Invoice #{invoice.id} — {currency} {invoice.total:.2f}",
                "body": html,
            })

            try:
                InvoiceRepository.update_status(invoice.id, "sent")
            except Exception as status_exc:
                logger.warning(
                    "Invoice #%s email sent but status update failed: %s",
                    invoice.id, status_exc,
                )

            return {
                "success": True,
                "message": f"Invoice #{invoice.id} sent to {recipient_email}",
                "email_result": email_result,
            }

        return {"success": False, "message": "Provide either a bill or an invoice_id to send"}

    except Exception as exc:
        return _safe_error("Invoice email delivery failed", exc=exc)
