# Tool-calling SFT dataset

Sample dataset for fine-tuning a single model (LFM2.5-VL) to do both frame
analysis *and* tool selection/intent classification, replacing the
prompt-engineered classifier in `agents/classifiers/llm_classifier.py`
(the big `_CLASSIFICATION_PROMPT` with the full tool table).

## Files

- `generate_dataset.py` — generator. Pulls tool schemas **live from the
  actual MCP tool registries** (`GeneralToolRegistry`, `PCBToolRegistry`)
  rather than hand-typing them, so the dataset can never drift out of sync
  with the real `input_schema` definitions in `tool_defs.py`. Example
  utterances and argument values are hand-authored in the `EXAMPLES` dict.
- `pcb_agent_tool_calls.jsonl` — generated output. Regenerate with:

  ```bash
  python3 data/tool_calling_sft/generate_dataset.py > data/tool_calling_sft/pcb_agent_tool_calls.jsonl
  ```

## Format

One JSON object per line, OpenAI-style function-calling:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Start monitoring the conveyor for defects."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_start_monitoring_session", "type": "function",
       "function": {"name": "start_monitoring_session", "arguments": "{\"description\": \"...\"}"}}
    ]}
  ],
  "tools": [ /* all 35 tool schemas, OpenAI function format */ ]
}
```

No-tool-call examples (greetings, thanks, out-of-scope chat) use a plain
`content` string on the assistant turn instead of `tool_calls` — the
domain is PCB-only, so anything outside that should get a conversational
reply, never a tool call.

## Coverage

35/35 registered tools (13 general + 22 PCB), 1-2 examples each, plus 10
no-tool-call negatives. **This is a bootstrap sample, not a production
SFT set** — expect to need several examples per tool (varied phrasing,
edge cases, multi-turn context) for the model to generalize well. Add to
`EXAMPLES` / `NO_TOOL_EXAMPLES` in `generate_dataset.py` and regenerate
rather than hand-editing the JSONL directly, so schemas stay authoritative.

## Fine-tuning LFM2.5-VL specifically

Checked the deployed model's own `chat_template.jinja`
(`LFM2.5-VL-1.6B-PCB-Inspect`): it injects a `tools` list into the system
prompt (so the model sees what's available) but has **no dedicated
template block for rendering `tool_calls` in assistant turns** — unlike
e.g. Qwen's `<tool_call>...</tool_call>` tags. That means the exact
on-the-wire format for an assistant tool call is established by whatever
convention your fine-tuning framework renders `tool_calls` into, not by
the base model's template. Check what your SFT framework (axolotl,
LLaMA-Factory, a custom script, etc.) expects before training — you may
need to convert the `arguments` field from a JSON string (used here, per
the OpenAI spec) into that framework's inline format.
