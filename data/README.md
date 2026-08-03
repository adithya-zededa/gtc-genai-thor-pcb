# `data/` — fine-tuning datasets

This folder holds training data for fine-tuning the vision-language model
used by the camera-agent, as opposed to the application code that serves
it. **You should not need to read the rest of this repository to work
here.** Everything you need — the full tool catalog, the dataset format,
and the gotchas specific to this model — is in this file and in
`tool_calling_sft/README.md`.

## What's in here

- **`tool_calling_sft/`** — a sample dataset teaching the model to select
  and call the right tool for a user's chat message (intent
  classification + function calling), instead of the app's current
  hand-written classifier prompt. See `tool_calling_sft/README.md` for
  the dataset itself.

## The model

Target model: **`LiquidAI/LFM2.5-VL-1.6B-PCB-Inspect`**, a fine-tune of
`LiquidAI/LFM2.5-VL-1.6B-Extract` for PCB defect inspection. 1.6B
parameters, vision-language, currently served via vLLM.

## Read this before you train anything

These are concrete, verified findings about this specific checkpoint —
not general fine-tuning advice. Each one will either break training
silently or damage a capability you need in production.

### 1. The shipped chat template cannot render OpenAI-style `tool_calls`

The tokenizer's vocabulary has real tool-calling special tokens inherited
from the base LFM2 line (`tool_call_start`, `tool_call_end`,
`tool_list_start`, `tool_list_end`, `tool_response_end`). But **this
checkpoint's `chat_template.jinja` never reads `message["tool_calls"]`** —
it only reads `message["content"]`. Tools passed via the `tools` kwarg
just get flattened into a `"List of tools: [...]"` string in the system
prompt; there's no template branch that renders an assistant tool call at
all.

If your fine-tuning framework calls `tokenizer.apply_chat_template()` on
this dataset as-is, every tool-calling example (`content: null,
tool_calls: [...]`) will render as an **empty assistant turn** — you'd
be training the model to say nothing whenever a tool call is warranted.

Before training: render one example through the actual tokenizer and
read the output. If it's blank, you need to either (a) find the
tool-call convention from a base LFM2 checkpoint that *does* use these
special tokens and pre-render assistant turns yourself before this
template touches them, or (b) confirm your SFT framework does its own
tool-call rendering and bypasses the shipped template.

### 2. This model is a narrow extractor, not a chat/agent model

From the model's own README:

> Trained for the schema style above; free-form VQA prompts are out of
> scope.

It was trained for exactly one shape: a component schema in the system
message, an image-only user turn, flat JSON enum output. This dataset is
the opposite on every axis — no images, conversational framing,
tool-calling semantics. Fine-tuning the *same* adapter weights on this
risks catastrophic forgetting of the PCB-inspection skill that production
depends on.

Don't stack a tool-calling LoRA onto the existing vision adapter. Prefer
one of:
- Merge the vision LoRA into base weights, then train a **separate**
  fresh LoRA for tool-calling.
- Serve two independent LoRA adapters (vision, tool-calling) via vLLM's
  multi-LoRA support and select per-request — neither training run ever
  touches the other's weights.

### 3. The system prompt here doesn't match any prompt already in production

There are three other system prompts already live in this codebase, and
none of them agree with each other or with what's in this dataset:
the model's own documented format, the app's current vision-inspection
prompt (`agents/vlm/prompts.py`), and the app's current text classifier
prompt (`agents/classifiers/llm_classifier.py`). If this dataset is meant
to replace the classifier prompt, whatever system prompt you settle on
for training must also be what the app sends at inference time — a
fine-tune trained against one prompt and served with another will
underperform.

### 4. Small dataset, high memorization risk

Every example currently carries the *identical* system prompt and the
*full* 35-tool schema block. With a small dataset and LoRA's limited
capacity, that's a strong incentive for the model to memorize "this exact
prompt string → tool-calling mode" rather than learn the general skill.
There are also no hard negatives between semantically close tools (e.g.
`get_defect_summary` vs. `count_defective_pcbs` vs.
`get_defect_type_breakdown`) — add some as you extend the dataset.

## Tool catalog (35 tools)

Every tool the fine-tuned model needs to be able to call, with its full
parameter schema. This is generated directly from the application's tool
registries — see "Keeping this catalog current" below — so treat it as
authoritative over anything else you're told about available tools.

<!--
TOOL_CATALOG_START
Regenerate this section with:
    python3 data/tool_calling_sft/print_tool_catalog.py
Paste its output between these markers, replacing the previous version.
-->

<!-- AUTO-GENERATED by data/tool_calling_sft/print_tool_catalog.py — do not hand-edit. -->
<!-- 13 general-domain tools, 22 PCB-domain tools, 35 total. -->

### General domain (13 tools)

#### `acknowledge_error`

*Category: `control`* · *Requires confirmation: no*

Acknowledge an error state and attempt to recover to idle.

*No parameters.*

#### `analyze_current_frame`

*Category: `analysis`* · *Requires confirmation: no*

Analyze the current camera frame and describe what is visible. Can include a specific question.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | no | — | Optional specific question about the frame (max length: 500) |

#### `end_session`

*Category: `session`* · *Requires confirmation: no*

End the current session. Monitoring will stop and a summary will be provided.

*No parameters.*

#### `get_agent_status`

*Category: `status`* · *Requires confirmation: no*

Get the current status of the monitoring agent including state, session info, and statistics.

*No parameters.*

#### `get_session_summary`

*Category: `session`* · *Requires confirmation: no*

Get a summary of the current or previous session including events and statistics.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `session_id` | string | no | — | Optional session ID. If not provided, summarizes the current/most recent session. |

#### `go_idle`

*Category: `control`* · *Requires confirmation: no*

Pause active monitoring and enter idle state. Camera monitoring will stop.

*No parameters.*

#### `log_event`

*Category: `logging`* · *Requires confirmation: no*

Log an event to the persistent audit log. Only use when the user explicitly requests logging — automatic DB logging handles routine persistence.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `event_type` | string (`observation` / `detection` / `alert` / `system` / `user_action`) | yes | — | Type of event |
| `description` | string | yes | — | Description of the event (max length: 1000) |
| `severity` | string (`info` / `warning` / `error`) | no | `info` | Event severity level |

#### `query_history`

*Category: `history`* · *Requires confirmation: no*

Query recent detection history and events.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | integer | no | `10` | Maximum number of events to return (range: 1–100) |
| `event_type` | string (`detection` / `alert` / `all`) | no | — | Filter by event type |

#### `save_evidence`

*Category: `evidence`* · *Requires confirmation: no*

Save the current frame as evidence for later review.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `label` | string | yes | — | Label to identify this evidence (max length: 100) |
| `notes` | string | no | — | Additional notes about the evidence (max length: 1000) |

#### `send_alert_email`

*Category: `alerts`* · *Requires confirmation: yes*

Send an alert email to specified recipients with optional image attachment.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `recipients` | array of string | no | — | Optional list of email addresses; if omitted, configured default recipients are used |
| `subject` | string | yes | — | Email subject line (max length: 200) |
| `body` | string | yes | — | Email body content (max length: 5000) |
| `include_image` | boolean | no | `True` | Whether to attach the current camera frame |
| `priority` | string (`low` / `normal` / `high`) | no | `normal` | Email priority level |

#### `set_detection_task`

*Category: `configuration`* · *Requires confirmation: no*

Configure what the agent should look for during monitoring.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `task_type` | string (`pcb_inspection` / `custom`) | yes | — | The type of detection task |
| `custom_instructions` | string | no | — | Custom instructions for the detection task (required for custom task type) (max length: 2000) |

#### `shutdown_agent`

*Category: `control`* · *Requires confirmation: yes*

Completely shut down the agent. All monitoring will stop.

*No parameters.*

#### `start_monitoring_session`

*Category: `session`* · *Requires confirmation: no*

Start a new monitoring session. The agent will begin actively watching the camera feed.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `description` | string | no | — | Optional description of what to monitor for (max length: 500) |

### PCB domain (22 tools)

#### `check_threshold_alerts`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Check whether any defects exceeded predefined thresholds or alert conditions. Use when user asks about threshold violations or alert conditions.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `rate_threshold` | number | no | — | Max acceptable defects per window (default: 10) |
| `window_hours` | number | no | — | Time window in hours (default: 24) |

#### `classify_board`

*Category: `pcb_analysis`* · *Requires confirmation: no*

Identify the type of PCB board visible in the current frame.

*No parameters.*

#### `count_defective_pcbs`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Count how many defective PCBs were detected overall or within a given time window. Use when user asks 'how many defects' or 'how many bad PCBs'.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Time window in hours. Omit for all-time count. |

#### `generate_defect_report`

*Category: `pcb_reporting`* · *Requires confirmation: no*

Generate a summary report of all recorded PCB defects.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `board_type` | string | no | — | Filter report to a specific board type |

#### `generate_summary_report`

*Category: `pcb_reporting`* · *Requires confirmation: no*

Generate a comprehensive daily, weekly, or on-demand summary report of monitoring results including defect statistics, trends, and threshold status.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `period` | string (`daily` / `weekly` / `all`) | no | `daily` | Report period |
| `board_type` | string | no | — | Filter to a specific board type |

#### `get_defect_insights`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get AI-driven recommendations and insights based on observed defect patterns. Use when user asks for analysis, recommendations, or 'what should we do'.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Analysis window in hours |

#### `get_defect_summary`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get a summary of defects identified within a specific time range, log, or board type. Use when user asks about defects in a period or for a specific source.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Time window in hours (e.g., 24 for last day, 168 for last week) |
| `board_type` | string | no | — | Filter by board type |
| `severity` | string (`low` / `medium` / `high`) | no | — | Filter by severity |

#### `get_defect_trend`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Analyze whether defect rates are increasing, decreasing, or stable over time. Use when user asks about trends or rate changes.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Analysis window in hours (default: 168 = 1 week) |

#### `get_defect_type_breakdown`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get what types of defects have been identified and how frequently each occurs. Use when user asks about defect categories, types, or frequencies.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Time window in hours |

#### `get_latest_defect`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get the most recently identified defect with full details. Use when user asks 'when was the last defect' or 'most recent defect'.

*No parameters.*

#### `get_latest_pcb_frames`

*Category: `pcb_analysis`* · *Requires confirmation: no*

Retrieve metadata for the most recently stored PCB frames. Frames are auto-captured by the monitoring pipeline when a PCB is detected with low motion. Use this to see what frames are available before inspecting them.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | integer | no | `5` | Max number of frames to return (range: 1–20) |
| `unconsumed_only` | boolean | no | `True` | Only return frames not yet inspected |

#### `get_monitoring_status`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get the current and historical monitoring status including whether monitoring is active, total defect counts, and recent activity summary.

*No parameters.*

#### `get_most_severe_defect`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get the most severe or highest-priority defect detected. Use when user asks about worst defect, highest severity, or most critical issue.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Time window in hours |

#### `get_notification_preferences`

*Category: `pcb_notifications`* · *Requires confirmation: no*

Get the current notification preferences and configuration. Use when user asks about their notification settings.

*No parameters.*

#### `get_top_defect_sources`

*Category: `pcb_analytics`* · *Requires confirmation: no*

Get which board types or sources produce the most defects. Use when user asks which boards have the most problems.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `hours` | number | no | — | Time window in hours |

#### `inspect_pcb`

*Category: `pcb_analysis`* · *Requires confirmation: no*

Inspect the current camera frame for PCB defects including solder bridges, missing components, trace damage, and more. Defects are auto-logged to the database — you do NOT need to call log_defect afterwards.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | no | — | Optional specific question about the PCB (max length: 500) |

#### `inspect_pcb_frame`

*Category: `pcb_analysis`* · *Requires confirmation: no*

Send a stored PCB frame to the VLM for defect inspection. Returns analysis so you can reason about next steps. By default this tool is analysis-only; set auto_log=true to persist detected defects. If frame_id is omitted, uses the most recent unconsumed frame.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `frame_id` | integer | no | — | ID of a stored frame from get_latest_pcb_frames. If omitted, uses the newest unconsumed frame. |
| `query` | string | no | — | Optional specific question about the PCB (max length: 500) |
| `auto_log` | boolean | no | `False` | If true, automatically log detected defects |

#### `log_defect`

*Category: `pcb_logging`* · *Requires confirmation: no*

Manually record a PCB defect to the persistent database. Use when the user explicitly requests manual logging, or when inspect_pcb_frame was run with auto_log=false.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `board_type` | string | yes | — | Board type |
| `defect_type` | string | yes | — | Defect classification |
| `severity` | string (`low` / `medium` / `high`) | no | `low` | Severity level |
| `confidence` | number | no | — | Detection confidence 0-1 (range: 0.0–1.0) |
| `description` | string | no | — | Human-readable defect description (max length: 2000) |

#### `query_detection_logs`

*Category: `pcb_reporting`* · *Requires confirmation: no*

Query detection logs enriched with defect information. Provides a unified view of detection events and defect records. Use when user asks to see logs, detection history, or what was inspected.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | integer | no | `20` | Max records to return (default 20) (range: 1–100) |
| `hours` | number | no | — | Time window in hours |
| `detected_only` | boolean | no | `False` | Only return detections with confidence > 0 |
| `include_defects` | boolean | no | `True` | Include defect records alongside detection logs |

#### `query_pcb_inspections`

*Category: `pcb_reporting`* · *Requires confirmation: no*

Query past PCB inspection results from the database. Use this to answer user questions about detected defects, pass rates, board types seen, and inspection history. Supports time-window filtering.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `limit` | integer | no | `10` | Max records to return (range: 1–50) |
| `result_filter` | string (`PASS` / `FAIL`) | no | — | Filter by result: PASS, FAIL, or omit for all |
| `hours` | number | no | — | Time window in hours (e.g., 24 for last day, 168 for last week) |

#### `send_defect_alert`

*Category: `pcb_alerts`* · *Requires confirmation: yes*

Send an email alert about a detected PCB defect to specified recipients. You should first inspect a frame using inspect_pcb_frame and only call this tool if you determined a defect is present.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `recipients` | array of string | yes | — | List of email addresses to alert |
| `board_type` | string | no | `unknown` | Board type (from inspection) |
| `defect_summary` | string | no | — | Description of the defect. If omitted, a summary is auto-generated from recent defect logs. (max length: 5000) |
| `severity` | string (`low` / `medium` / `high`) | no | `medium` | Defect severity |
| `include_image` | boolean | no | `True` | Attach the PCB frame image |

#### `toggle_email_notifications`

*Category: `pcb_notifications`* · *Requires confirmation: no*

Enable or disable email notifications for detected defects. Can also set minimum severity threshold and configure recipient list.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `enabled` | boolean | no | — | True to enable, False to disable. Omit to toggle. |
| `min_severity` | string (`low` / `medium` / `high`) | no | — | Minimum severity to trigger notifications |
| `recipients` | array of string | no | — | Email addresses for notifications |

<!-- TOOL_CATALOG_END -->

## Dataset format

One JSON object per line (JSONL), OpenAI-style function calling:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Start monitoring the conveyor for defects."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_start_monitoring_session",
          "type": "function",
          "function": {
            "name": "start_monitoring_session",
            "arguments": "{\"description\": \"Watch for PCB defects on the conveyor\"}"
          }
        }
      ]
    }
  ],
  "tools": [ /* full tool schemas, see print_tool_catalog.py output above */ ]
}
```

`function.arguments` is a **JSON-encoded string**, not a nested object —
that's the OpenAI convention this format follows. Examples that shouldn't
trigger a tool call (greetings, thanks, out-of-scope chat — this system
is PCB-inspection-only) use a plain string `content` on the assistant
turn instead of `tool_calls`.

## Adding examples

Don't hand-edit the `.jsonl` file. Add entries to the `EXAMPLES` or
`NO_TOOL_EXAMPLES` dict in `tool_calling_sft/generate_dataset.py` and
regenerate:

```bash
python3 data/tool_calling_sft/generate_dataset.py > data/tool_calling_sft/pcb_agent_tool_calls.jsonl
```

The generator validates every tool name in `EXAMPLES` against the live
registry and fails loudly on typos or renamed tools — that guarantee is
lost the moment you edit the JSONL directly.

When adding examples, prioritize:
- Tools with only 1 example today — most tools have just one or two;
  more phrasing variation per tool will generalize better.
- Hard negatives — pairs of similar user messages that must map to
  *different* tools (or to no tool), especially among the PCB analytics
  tools listed above, which overlap semantically.
- Multi-turn context if your target training format supports it — the
  live chat interface has follow-up turns (e.g. "how about last week?"
  after a defect-count question); this dataset is single-turn only.

## Keeping the tool catalog current

The catalog in this file is generated, not hand-written. Whenever a tool
is added, removed, or its schema changes in the application code,
regenerate it and paste the output back into the marked section above:

```bash
python3 data/tool_calling_sft/print_tool_catalog.py
```
