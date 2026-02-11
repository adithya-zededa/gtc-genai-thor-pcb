"""Invoice generation and formatting service for retail domain.

Handles business logic for invoice HTML generation, speech text formatting,
and invoice resending workflows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.logging import get_logger

logger = get_logger(__name__)


def format_invoice_for_speech(
    invoice_result: Dict[str, Any],
    bill: Optional[Dict[str, Any]] = None,
    *,
    full_narration: bool = False,
) -> Optional[str]:
    """Build a human-friendly spoken summary from an invoice result.

    Business logic for text composition, currency formatting, and narration style.

    Parameters
    ----------
    invoice_result : dict
        The result dict returned by invoice generation.
    bill : dict, optional
        The underlying bill data (line_items, totals, etc.).
    full_narration : bool
        When True, read every line item instead of summarizing.

    Returns
    -------
    str | None
        The spoken text, or None if insufficient data.
    """
    data = invoice_result.get("data", {})
    invoice_id = data.get("invoice_id")
    total = data.get("total", 0)
    recipient = data.get("recipient", "")

    currency = os.getenv("RETAIL_CURRENCY", "USD")
    currency_spoken = {
        "INR": "rupees",
        "USD": "dollars",
        "EUR": "euros",
        "GBP": "pounds",
    }.get(currency, currency)

    if invoice_id is None and total == 0:
        return None

    # Format the total for natural speech
    total_str = f"{total:,.2f}"
    total_spoken = f"{total_str} {currency_spoken}"

    lines: List[str] = []
    lines.append("Hello, here is your invoice summary.")

    if invoice_id:
        lines.append(f"Invoice number {invoice_id}.")

    if full_narration and bill:
        # Speak every line item
        line_items = bill.get("line_items", [])
        if line_items:
            lines.append("Here are the items on your invoice.")
            for item in line_items:
                name = item.get("name", "Unknown item")
                qty = item.get("quantity", 1)
                price = item.get("price", 0)
                lines.append(
                    f"{name}, quantity {qty}, at {price:,.2f} {currency_spoken} each."
                )

        subtotal = bill.get("subtotal")
        if subtotal is not None:
            lines.append(f"Subtotal: {subtotal:,.2f} {currency_spoken}.")
        tax = bill.get("tax")
        if tax is not None:
            lines.append(f"Tax: {tax:,.2f} {currency_spoken}.")

    lines.append(f"The total amount due is {total_spoken}.")

    if recipient:
        lines.append(f"This invoice has been prepared for {recipient}.")

    lines.append("Thank you for your business. Have a great day.")

    text = " ".join(lines)

    # Enforce max length
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    return text


def resend_invoice(invoice_id: int, recipient_email: str) -> Dict[str, Any]:
    """Fetch an existing invoice, regenerate HTML, and send via email.

    Business workflow for invoice resending, including data reconstruction,
    HTML generation, email delivery, and status updates.

    Parameters
    ----------
    invoice_id : int
        ID of the existing invoice.
    recipient_email : str
        Email address to send to.

    Returns
    -------
    dict
        Standard result dict with success, message.
    """
    from app.database.repositories import InvoiceRepository
    from services.domains.retail.service import generate_invoice_html
    from agents.tools.email import send_email

    # Fetch invoice
    invoice = InvoiceRepository.get_by_id(invoice_id)
    if not invoice:
        return {
            "success": False,
            "message": f"Invoice #{invoice_id} not found",
        }

    # Reconstruct bill data from DB model
    currency = os.getenv("RETAIL_CURRENCY", "USD")
    bill_data = {
        "line_items": invoice.items,
        "subtotal": invoice.subtotal,
        "tax_rate": (invoice.tax / invoice.subtotal) if invoice.subtotal else 0,
        "tax": invoice.tax,
        "total": invoice.total,
        "currency": currency,
        "generated_at": invoice.timestamp or "",
    }

    # Generate HTML
    html = generate_invoice_html(bill_data, recipient_email, invoice.id)

    # Send email
    email_result = send_email({
        "to": [recipient_email],
        "subject": f"Invoice #{invoice.id} — {currency} {invoice.total:.2f}",
        "body": html,
    })

    # Update status
    try:
        InvoiceRepository.update_status(invoice.id, "sent")
    except Exception as status_exc:
        logger.warning(
            "Invoice #%s email sent but status update failed: %s",
            invoice.id, status_exc,
        )

    return {
        "success": True,
        "message": f"✅ Invoice #{invoice.id} sent successfully to {recipient_email}. Total: {currency} {invoice.total:.2f}",
        "invoice_id": invoice.id,
        "email_result": email_result,
    }
