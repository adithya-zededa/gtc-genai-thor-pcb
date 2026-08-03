#!/usr/bin/env bash
# Refresh mcp/ from the application's agents/mcp/ source.
#
# mcp/ is a snapshot, not a live import — it exists so this directory
# can be handed to someone without the rest of the application code (see
# README.md, "Standalone package"). Snapshots drift. Run this from a full
# checkout of the application repository whenever schema.py,
# state_machine.py, registry.py, or either domain's tool_defs.py changes,
# then commit the result.
#
# Usage (from the application repo root):
#   data/tool_calling_sft/sync_registry.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="$SCRIPT_DIR/mcp"

if [ ! -d "$REPO_ROOT/agents/mcp" ]; then
    echo "error: $REPO_ROOT/agents/mcp not found — run this from within the application repo." >&2
    exit 1
fi

cp "$REPO_ROOT/agents/mcp/schema.py" "$DEST/schema.py"
cp "$REPO_ROOT/agents/mcp/state_machine.py" "$DEST/state_machine.py"
cp "$REPO_ROOT/agents/mcp/registry.py" "$DEST/registry.py"
cp "$REPO_ROOT/agents/mcp/domains/general/tool_defs.py" "$DEST/domains/general/tool_defs.py"
cp "$REPO_ROOT/agents/mcp/domains/pcb/tool_defs.py" "$DEST/domains/pcb/tool_defs.py"

# Rewrite the two import lines that only make sense inside the full
# application package tree, so the copies work standalone.
sed -i.bak 's/^from core\.logging import get_logger$/from ._logging import get_logger/' \
    "$DEST/state_machine.py" "$DEST/registry.py"
sed -i.bak \
    -e 's/^from agents\.mcp\.schema import/from mcp.schema import/' \
    -e 's/^from agents\.mcp\.state_machine import/from mcp.state_machine import/' \
    -e 's/^from agents\.mcp\.registry import/from mcp.registry import/' \
    "$DEST/domains/general/tool_defs.py" "$DEST/domains/pcb/tool_defs.py"
rm -f "$DEST"/*.bak "$DEST"/domains/*/*.bak

python3 "$SCRIPT_DIR/generate_dataset.py" > "$SCRIPT_DIR/pcb_agent_tool_calls.jsonl"
python3 "$SCRIPT_DIR/print_tool_catalog.py"

echo "mcp/ refreshed. Paste the print_tool_catalog.py output above into README.md" \
     "between the TOOL_CATALOG_START / TOOL_CATALOG_END markers, then review" \
     "the diff in pcb_agent_tool_calls.jsonl and mcp/ before committing."
