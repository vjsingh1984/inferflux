#!/usr/bin/env bash
# install-hooks.sh: configure git to use project-level hooks in .githooks/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.githooks"

if [ ! -d "$HOOKS_DIR" ]; then
  echo "ERROR: $HOOKS_DIR not found. Run this script from the repo root."
  exit 1
fi

# Make hooks executable
chmod +x "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/commit-msg"

# Wire git to use project hooks directory
git -C "$REPO_ROOT" config core.hooksPath .githooks

echo "Hooks installed. Project hooks in .githooks/ are now active."
echo ""
echo "  pre-commit : clang-format (if installed), cmake build, unit tests, stub integration tests"
echo "  commit-msg : blocks AI agent attribution and co-author trailers"
echo ""
echo "To uninstall, run:"
echo "  git config --unset core.hooksPath"
