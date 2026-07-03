#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_ROOT="$REPO_ROOT/embodied_gen/skills/claude_plugin"
PLUGIN_PACKAGE_ROOT="$PLUGIN_ROOT/plugins/embodiedgen"
CLAUDE_BIN="claude"
MARKETPLACE_NAME="embodiedgen-local"
PLUGIN_NAME="embodiedgen"

if [[ "${1:-}" != "" ]]; then
  echo "Usage: bash install/install_agent_plugin.sh" >&2
  exit 1
fi

if [[ ! -f "$PLUGIN_PACKAGE_ROOT/.claude-plugin/plugin.json" ]]; then
  echo "Plugin manifest not found: $PLUGIN_PACKAGE_ROOT/.claude-plugin/plugin.json" >&2
  exit 1
fi

if [[ ! -f "$PLUGIN_ROOT/.claude-plugin/marketplace.json" ]]; then
  echo "Marketplace manifest not found: $PLUGIN_ROOT/.claude-plugin/marketplace.json" >&2
  exit 1
fi

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "Claude CLI not found in PATH: $CLAUDE_BIN" >&2
  exit 1
fi

echo "Adding local Claude marketplace from:"
echo "  $PLUGIN_ROOT"
echo
if "$CLAUDE_BIN" plugin marketplace list | grep -q "$MARKETPLACE_NAME"; then
  echo "Marketplace already exists, updating:"
  echo "  $CLAUDE_BIN plugin marketplace update \"$MARKETPLACE_NAME\""
  "$CLAUDE_BIN" plugin marketplace update "$MARKETPLACE_NAME"
else
  echo "Running:"
  echo "  $CLAUDE_BIN plugin marketplace add \"$PLUGIN_ROOT\" --scope local"
  "$CLAUDE_BIN" plugin marketplace add "$PLUGIN_ROOT" --scope local
fi

echo
echo "Installing plugin:"
echo "  $CLAUDE_BIN plugin install \"$PLUGIN_NAME@$MARKETPLACE_NAME\" --scope local"
"$CLAUDE_BIN" plugin install "$PLUGIN_NAME@$MARKETPLACE_NAME" --scope local
