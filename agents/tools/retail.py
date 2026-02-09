"""Retail billing domain tools for the MCP agent.

Provides thin-adapter tool handler functions used by the Retail MCP executor.
Each function validates its inputs, delegates to the service layer,
and returns a sanitized result dict.

Tools:
- scan_tray_items: Analyze current frame to identify items on a tray
- lookup_item_price: Look up item prices from the retail catalog DB
- create_bill: Assemble a bill from detected/looked-up items
- generate_invoice: Render an HTML invoice from the current bill
- send_invoice_email: Email the invoice to a recipient
- speak_invoice: Convert a finalized invoice into spoken audio (TTS)
"""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from agents.tools.validation import (
    validate_email as _validate_email,
    safe_error as _safe_error,
    clamp_quantity as _clamp_quantity,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_scan_tray_items(
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current camera frame to identify and count retail items.

    Uses the VLM with ``RETAIL_BILLING`` task type.

    This tool handles structure extraction from VLM output but delegates
    validation to the service layer.
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

    # Extract structured items from VLM response
    # This is adapter logic: transform external format (VLM JSON) to internal format
    items: List[Dict[str, Any]] = []
    total_count = 0

    if event.full_response and isinstance(event.full_response, str):
        try:
            parsed = json.loads(event.full_response)
            raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []

            # Basic quantity normalization (not validation — just type safety)
            for it in raw_items:
                if "quantity" in it:
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
    """Look up an item's price from the retail catalog database.

    Thin adapter over service lookup methods with input sanitization.
    """
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

    Thin adapter that delegates all business logic to the service layer.

    Parameters
    ----------
    items : list of dict, optional
        Explicit items with name, quantity, price.
    scan_result : dict, optional
        Scan data from tool_scan_tray_items.

    Returns
    -------
    dict
        Bill with line_items, totals, warnings (if any).
    """
    logger.info(
        "tool_create_bill invoked (explicit_items=%s, has_scan=%s)",
        bool(items), bool(scan_result),
    )

    from services.domains.retail.bill_service import (
        create_bill_from_scan,
        create_bill_from_items,
    )

    try:
        # Path A: Create from scan result
        if scan_result:
            return create_bill_from_scan(scan_result)

        # Path B: Create from explicit items
        if items:
            return create_bill_from_items(items)

        # No input provided
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
    """Generate an HTML invoice from a bill and save it to the database.

    Validates email input and delegates to service layer.
    """
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

    Thin adapter that delegates to service layer workflows.

    Parameters
    ----------
    recipient_email : str
        Email address to send to.
    bill : dict, optional
        If provided, a new invoice is created, saved, and emailed.
    invoice_id : int, optional
        If provided, the existing invoice is fetched and emailed.

    Returns
    -------
    dict
        Result with success, message, email_result.
    """
    logger.info(
        "tool_send_invoice_email invoked (email=%s, has_bill=%s, invoice_id=%s)",
        recipient_email, bool(bill), invoice_id,
    )

    # Validate recipient
    err = _validate_email(recipient_email)
    if err:
        return {"success": False, "message": err}

    from services.domains.retail.service import save_and_send_invoice
    from services.domains.retail.invoice_service import resend_invoice

    try:
        # Path A: Create new invoice from bill and send
        if bill:
            return save_and_send_invoice(bill, recipient_email)

        # Path B: Resend existing invoice
        if invoice_id:
            return resend_invoice(invoice_id, recipient_email)

        return {"success": False, "message": "Provide either a bill or an invoice_id to send"}

    except Exception as exc:
        return _safe_error("Invoice email delivery failed", exc=exc)


def tool_speak_invoice(text: str = "") -> Dict[str, Any]:
    """Convert text to spoken audio and play it back.

    Uses Google Text-to-Speech (gTTS) and the infrastructure audio player.

    Parameters
    ----------
    text : str
        A spoken-friendly invoice summary (typically from the service layer).

    Returns
    -------
    dict
        Standard result dict with audio_file path if successful.
    """
    logger.info("tool_speak_invoice invoked (text_len=%d)", len(text))

    if not text or not text.strip():
        return {"success": False, "message": "No text provided for speech synthesis."}

    # Enforce max length
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    try:
        from gtts import gTTS
    except ImportError:
        return {
            "success": False,
            "message": "TTS dependency 'gtts' is not installed. Run: pip install gtts",
        }

    audio_path: Optional[str] = None
    try:
        tts = gTTS(text=text, lang="en", slow=False)

        # Write to a temporary mp3 file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", prefix="invoice_tts_", delete=False,
        )
        audio_path = tmp.name
        tmp.close()
        tts.save(audio_path)
        logger.info("TTS audio saved to %s", audio_path)

        # Use infrastructure service for playback
        from services.infrastructure.audio import AudioPlayer
        AudioPlayer().play_file(audio_path)

        return {
            "success": True,
            "message": "Invoice audio summary played successfully.",
            "data": {"audio_file": audio_path, "text_length": len(text)},
        }

    except Exception as exc:
        logger.error("TTS synthesis/playback failed: %s", exc, exc_info=True)
        return _safe_error("Text-to-speech failed", exc=exc)
