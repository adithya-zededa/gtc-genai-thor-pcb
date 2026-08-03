#!/usr/bin/env python3
"""Render every example in the dataset through a real tokenizer.

generate_dataset.py cannot verify its own output is renderable — that
requires the actual model tokenizer, which is not part of this
standalone directory (see README.md). This script fills that gap: point
it at a local checkpoint (or a HF repo id, if you have network access
and the model isn't gated) and it calls tokenizer.apply_chat_template()
on every example, reporting any that fail.

Requires `transformers` (not a dependency of the rest of this
directory — install separately, e.g. in a venv):
    pip install transformers

Usage:
    python3 validate_render.py /path/to/local/checkpoint
    python3 validate_render.py LiquidAI/LFM2-1.2B-Tool
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tokenizer path or HF repo id>", file=sys.stderr)
        raise SystemExit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is required: pip install transformers", file=sys.stderr)
        raise SystemExit(1)

    tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)

    jsonl_path = Path(__file__).resolve().parent / "pcb_agent_tool_calls.jsonl"
    examples = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                examples.append(json.loads(line))

    failures = []
    for i, example in enumerate(examples):
        try:
            tok.apply_chat_template(example["messages"], tools=example["tools"], tokenize=False)
        except Exception as exc:  # noqa: BLE001
            failures.append((i, example["messages"][1]["content"], exc))

    print(f"{len(examples)} examples rendered, {len(failures)} failed")
    for i, utterance, exc in failures:
        print(f"  [{i}] {utterance!r}: {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
