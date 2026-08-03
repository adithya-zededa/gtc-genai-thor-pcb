#!/usr/bin/env python3
"""Generate a tool-calling SFT dataset for the PCB inspection agent.

Pulls tool schemas live from the actual MCP tool registries (never
hand-typed) and pairs them with hand-authored example utterances to
produce OpenAI-style function-calling JSONL:

    {"messages": [system, user, assistant], "tools": [...]}

The assistant message either contains a "tool_calls" entry (for
examples that should trigger a tool) or plain text content (for the
no-tool-needed negative examples).

Intended for fine-tuning a single model (LFM2.5-VL) to do both frame
analysis *and* tool selection/intent classification, replacing the
current prompt-engineered classifier in agents/classifiers/llm_classifier.py.

Imports the tool registries from the local mcp/ package (a trimmed
snapshot of agents/mcp/, containing only the schema-definition files —
see README.md). This script has no dependency on the rest of the
application; it only needs this directory.

Usage:
    python3 data/tool_calling_sft/generate_dataset.py \
        > data/tool_calling_sft/pcb_agent_tool_calls.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.domains.general.tool_defs import GeneralToolRegistry  # noqa: E402
from mcp.domains.pcb.tool_defs import PCBToolRegistry  # noqa: E402


SYSTEM_PROMPT = (
    "You are the ZEDEDA PCB Inspection Agent, monitoring an Arduino Uno R4 "
    "Minima conveyor line via camera. Decide whether the user's message "
    "needs a tool call. If it does, call exactly one tool with the correct "
    "arguments. If it's a greeting, thanks, or something outside PCB "
    "monitoring, reply in plain text instead — do not call a tool."
)

# tool_name -> list of (user_utterance, arguments)
EXAMPLES: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    # ---- general domain ----
    "start_monitoring_session": [
        ("Start monitoring the conveyor for defects.",
         {"description": "Watch for PCB defects on the conveyor"}),
        ("Begin watching the camera.", {}),
    ],
    "end_session": [
        ("Stop monitoring now.", {}),
        ("End the session please.", {}),
    ],
    "get_session_summary": [
        ("Can you summarize this session?", {}),
        ("Give me a recap of session session_4a823c3c63d2.",
         {"session_id": "session_4a823c3c63d2"}),
    ],
    "get_agent_status": [
        ("What's your current status?", {}),
        ("Are you running right now?", {}),
    ],
    "analyze_current_frame": [
        ("What do you see right now?", {"query": "What do you see right now?"}),
        ("Describe the current camera view.", {"query": "Describe the current camera view"}),
    ],
    "send_alert_email": [
        ("Email john@example.com about the defect we just found, mark it high priority.",
         {"recipients": ["john@example.com"], "subject": "Defect Alert",
          "body": "A defect was just detected on the inspected PCB.",
          "priority": "high", "include_image": True}),
    ],
    "save_evidence": [
        ("Save this frame as evidence labeled 'suspicious solder joint'.",
         {"label": "suspicious solder joint"}),
        ("Save the current frame as evidence, call it 'board misalignment' and note it happened during the night shift.",
         {"label": "board misalignment", "notes": "Observed during night shift"}),
    ],
    "log_event": [
        ("Log that we detected motion on the line.",
         {"event_type": "detection", "description": "Motion detected on the conveyor line", "severity": "info"}),
        ("Record an error event: the camera disconnected briefly.",
         {"event_type": "system", "description": "Camera disconnected briefly", "severity": "error"}),
    ],
    "query_history": [
        ("Show me the last 10 detections.", {"limit": 10, "event_type": "detection"}),
        ("What's happened recently?", {"event_type": "all"}),
    ],
    "set_detection_task": [
        ("Switch to PCB inspection mode.", {"task_type": "pcb_inspection"}),
        ("Set a custom task: watch specifically for missing capacitors.",
         {"task_type": "custom", "custom_instructions": "Watch specifically for missing capacitors"}),
    ],
    "go_idle": [
        ("Take a break for now.", {}),
        ("Go idle.", {}),
    ],
    "shutdown_agent": [
        ("Shut yourself down.", {}),
    ],
    "acknowledge_error": [
        ("I acknowledge the error, please continue.", {}),
        ("Clear the error state.", {}),
    ],

    # ---- PCB domain ----
    "get_latest_pcb_frames": [
        ("Show me the latest PCB frames captured.", {"limit": 5}),
        ("List the frames that haven't been reviewed yet.", {"unconsumed_only": True}),
    ],
    "inspect_pcb_frame": [
        ("Inspect frame 42 for defects.", {"frame_id": 42}),
        ("Check frame 17 for solder issues and log it automatically.",
         {"frame_id": 17, "query": "Check for solder issues", "auto_log": True}),
    ],
    "inspect_pcb": [
        ("Inspect this board for defects.", {}),
        ("Check this PCB for missing components.", {"query": "Check for missing components"}),
    ],
    "classify_board": [
        ("What kind of board is this?", {}),
    ],
    "send_defect_alert": [
        ("Email quality@zededa.com about the solder bridge defect on the Arduino board, high severity.",
         {"recipients": ["quality@zededa.com"], "board_type": "Arduino Uno R4 Minima",
          "defect_summary": "Solder bridge detected between adjacent pads",
          "severity": "high", "include_image": True}),
    ],
    "log_defect": [
        ("Log a defect: Arduino board, bent header pin, medium severity.",
         {"board_type": "Arduino Uno R4 Minima", "defect_type": "bent header pin", "severity": "medium"}),
        ("Record a high severity solder bridge defect with 0.92 confidence.",
         {"board_type": "Arduino Uno R4 Minima", "defect_type": "solder bridge",
          "severity": "high", "confidence": 0.92}),
    ],
    "generate_defect_report": [
        ("Generate a defect report.", {}),
        ("Give me a defect report for Arduino boards.", {"board_type": "Arduino Uno R4 Minima"}),
    ],
    "query_pcb_inspections": [
        ("How many boards failed inspection in the last 24 hours?",
         {"result_filter": "FAIL", "hours": 24}),
        ("Show me the last 20 inspections.", {"limit": 20}),
    ],
    "get_monitoring_status": [
        ("Is monitoring currently running?", {}),
    ],
    "toggle_email_notifications": [
        ("Turn on email notifications for high severity defects.",
         {"enabled": True, "min_severity": "high"}),
        ("Disable email alerts.", {"enabled": False}),
    ],
    "get_defect_summary": [
        ("Summarize today's defects.", {"hours": 24}),
        ("Give me a summary of high severity defects this week.", {"hours": 168, "severity": "high"}),
    ],
    "count_defective_pcbs": [
        ("How many defective PCBs today?", {"hours": 24}),
    ],
    "get_latest_defect": [
        ("When was the most recent defect detected?", {}),
    ],
    "get_defect_type_breakdown": [
        ("What types of defects have we seen?", {}),
        ("Break down defect types over the last week.", {"hours": 168}),
    ],
    "get_defect_trend": [
        ("Are defects trending up or down?", {}),
    ],
    "get_most_severe_defect": [
        ("What's the worst defect we've found?", {}),
    ],
    "get_top_defect_sources": [
        ("Which boards have the most defects?", {}),
    ],
    "generate_summary_report": [
        ("Generate today's summary report.", {"period": "daily"}),
        ("Give me the weekly report.", {"period": "weekly"}),
    ],
    "check_threshold_alerts": [
        ("Are we within our defect rate thresholds?", {}),
        ("Check if the defect rate exceeded 5% in the last 12 hours.",
         {"rate_threshold": 0.05, "window_hours": 12}),
    ],
    "get_defect_insights": [
        ("What insights do you have based on recent defects?", {}),
    ],
    "get_notification_preferences": [
        ("What are my current notification settings?", {}),
    ],
    "query_detection_logs": [
        ("Show me detection logs from the last 6 hours.", {"hours": 6}),
        ("List only the logs where something was detected, with defect details.",
         {"detected_only": True, "include_defects": True}),
    ],
}

# No-tool-call negative examples: greetings, thanks, out-of-scope chat.
# The classifier prompt this dataset replaces is explicit that these must
# map to domain=general, tool=null.
NO_TOOL_EXAMPLES: List[Tuple[str, str]] = [
    ("Hey, how's it going?", "Hi! I'm watching the PCB line right now. Let me know if you'd like a status update or want me to inspect a board."),
    ("Thanks for the help!", "You're welcome — happy to keep watching the line for you."),
    ("Who are you?", "I'm the ZEDEDA PCB Inspection Agent. I monitor the conveyor camera and can inspect boards, log defects, and send alerts on request."),
    ("What's the weather like today?", "I don't have access to weather data — I'm scoped to PCB inspection and monitoring on this line."),
    ("Can you write me a poem?", "That's outside what I can help with here — I'm set up specifically for PCB defect monitoring and reporting."),
    ("Good morning!", "Good morning! Let me know if you'd like a status check or a frame analysis."),
    ("Cool, that makes sense.", "Glad that helped — anything else you'd like me to check on the line?"),
    ("What can you do?", "I can inspect PCBs for defects, log and report on them, manage monitoring sessions, and send email alerts — just ask."),
    ("Never mind.", "No problem — I'll keep monitoring in the meantime."),
    ("Is this thing on?", "Yes, I'm here and connected to the camera feed."),
]


def build_tool_schema(tool) -> Dict[str, Any]:
    """OpenAI function-calling tool schema, trimmed to the standard fields."""
    schema = tool.to_json_schema()
    return {
        "type": "function",
        "function": {
            "name": schema["function"]["name"],
            "description": schema["function"]["description"],
            "parameters": schema["function"]["parameters"],
        },
    }


def main() -> None:
    general_tools = GeneralToolRegistry().list_tools()
    pcb_tools = PCBToolRegistry().list_tools()
    all_tools = general_tools + pcb_tools
    tools_by_name = {t.name: t for t in all_tools}

    tool_schemas = [build_tool_schema(t) for t in all_tools]

    missing = set(EXAMPLES) - set(tools_by_name)
    if missing:
        raise SystemExit(f"EXAMPLES references unknown tool(s): {sorted(missing)}")

    unlabeled = set(tools_by_name) - set(EXAMPLES)
    if unlabeled:
        print(f"# NOTE: no examples for: {sorted(unlabeled)}", file=sys.stderr)

    records: List[Dict[str, Any]] = []

    for tool_name, examples in EXAMPLES.items():
        for utterance, arguments in examples:
            records.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": utterance},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{tool_name}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                ],
                "tools": tool_schemas,
            })

    for utterance, reply in NO_TOOL_EXAMPLES:
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": utterance},
                {"role": "assistant", "content": reply},
            ],
            "tools": tool_schemas,
        })

    for record in records:
        print(json.dumps(record, ensure_ascii=False))

    print(
        f"# Generated {len(records)} examples "
        f"({len(records) - len(NO_TOOL_EXAMPLES)} tool calls, {len(NO_TOOL_EXAMPLES)} no-tool) "
        f"covering {len(EXAMPLES)}/{len(tools_by_name)} tools",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
