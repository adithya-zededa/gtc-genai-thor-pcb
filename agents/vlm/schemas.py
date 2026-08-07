"""JSON Schemas for structured VLM decoding.

These constrain the model's output grammar so a malformed response is
impossible, replacing best-effort regex extraction in
``UnifiedVLMClient._parse_json_response``.

A schema must match the JSON block described in the corresponding prompt in
``agents.vlm.prompts``. If a prompt's response block changes, change the
schema with it — an under-specified schema is worse than none, because
structured decoding will happily constrain away fields the UI depends on.

Served through vLLM's OpenAI-compatible ``response_format`` field. Note that
the older ``guided_json`` request field is *silently ignored* by vLLM 0.24
(verified against the deployed server): it returns unconstrained output with
no error, so ``response_format`` is the only safe way to request this.
"""

from __future__ import annotations

from typing import Any, Dict


_COMPONENT_STATUS = {
    "type": "string",
    "enum": ["present_intact", "missing", "damaged", "uncertain"],
}

# Bounds on the free-form fields. Without them, constrained decoding on the
# deployed model is bimodal: usually ~92 completion tokens, but with a
# non-trivial rate of running away until it hits max_tokens (measured at 2048
# tokens / ~20s). Capping max_tokens alone only converts that stall into a
# truncated, unparseable response; bounding the grammar stops it happening.
# Measured over 10 runs each: unbounded worst case 3145 ms (and 2048-token
# runaways in other batches) vs bounded worst case 1127 ms, 10/10 valid.
_MAX_REASONING_CHARS = 200
_MAX_DEFECTS = 6

# Backstop only — with the bounds above the model stops on its own at ~95
# tokens. This exists so a pathological run costs ~6s instead of ~26s.
STRUCTURED_MAX_TOKENS = 512


# Matches DEFAULT_MONITORING_DEFECT_PROMPT's response block. ``defects`` and
# the three component-status fields are what templates/logs.html renders, so
# they are required rather than optional.
PCB_INSPECTION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "details": {
            "type": "object",
            "properties": {
                "power_jack_status": _COMPONENT_STATUS,
                "usb_port_status": _COMPONENT_STATUS,
                "header_pins_status": _COMPONENT_STATUS,
                "defects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": _MAX_DEFECTS,
                },
            },
            "required": [
                "power_jack_status",
                "usb_port_status",
                "header_pins_status",
                "defects",
            ],
            "additionalProperties": False,
        },
        "detected": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": _MAX_REASONING_CHARS},
        "should_alert": {"type": "boolean"},
    },
    "required": [
        "details",
        "detected",
        "confidence",
        "reasoning",
        "should_alert",
    ],
    "additionalProperties": False,
}


# Matches CUSTOM_QUERY_TEMPLATE, used for ad-hoc queries where the shape of
# ``details`` is dictated by the caller's prompt rather than known here — so
# it is left unconstrained and optional.
GENERIC_ANALYSIS_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "detected": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string", "maxLength": _MAX_REASONING_CHARS},
        "should_alert": {"type": "boolean"},
        "details": {"type": "object"},
    },
    "required": ["detected", "confidence", "reasoning", "should_alert"],
}


def schema_for_prompt(prompt: str) -> Dict[str, Any]:
    """Pick the schema whose response block matches *prompt*.

    The PCB inspection prompt is recognised by the component-status fields it
    asks for; anything else falls back to the generic analysis shape.
    """
    if "power_jack_status" in prompt and "usb_port_status" in prompt:
        return PCB_INSPECTION_RESPONSE_SCHEMA
    return GENERIC_ANALYSIS_RESPONSE_SCHEMA
