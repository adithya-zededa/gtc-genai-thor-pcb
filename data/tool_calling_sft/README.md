# `tool_calling_sft/`

Sample dataset for fine-tuning LFM2.5-VL to do both frame analysis and
tool selection/intent classification, replacing the prompt-engineered
classifier in `agents/classifiers/llm_classifier.py`.

**Start at [`../README.md`](../README.md)** — it has the full tool
catalog, the dataset format spec, and a "read before you train" section
covering four concrete issues with this specific model checkpoint
(a chat-template gap that can silently blank out every tool-call example,
catastrophic-forgetting risk, prompt mismatches, and memorization risk).
This file only covers what's specific to the scripts below.

## Files

- **`generate_dataset.py`** — generates `pcb_agent_tool_calls.jsonl`.
  Pulls tool schemas **live** from the actual MCP tool registries
  (`GeneralToolRegistry`, `PCBToolRegistry`) rather than hand-typing
  them, and validates every tool name in its `EXAMPLES` dict against the
  live registry — it fails loudly on typos or renamed tools. Example
  utterances and argument values are hand-authored in `EXAMPLES` /
  `NO_TOOL_EXAMPLES`.
- **`print_tool_catalog.py`** — generates the tool catalog markdown
  embedded in `../README.md`. Same live-schema guarantee as above.
- **`pcb_agent_tool_calls.jsonl`** — generated output. Never hand-edit
  this file — edit `EXAMPLES`/`NO_TOOL_EXAMPLES` in `generate_dataset.py`
  and regenerate.

## Regenerating

```bash
# Dataset
python3 data/tool_calling_sft/generate_dataset.py > data/tool_calling_sft/pcb_agent_tool_calls.jsonl

# Tool catalog (paste output into ../README.md between the
# TOOL_CATALOG_START / TOOL_CATALOG_END markers, replacing what's there)
python3 data/tool_calling_sft/print_tool_catalog.py
```

## Coverage

35/35 registered tools (13 general + 22 PCB), 1-2 examples each, plus 10
no-tool-call negatives. **This is a bootstrap sample, not a production
SFT set** — see "Adding examples" in `../README.md` for what to prioritize
when extending it.
