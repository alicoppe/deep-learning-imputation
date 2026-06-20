#!/usr/bin/env bash
# Make xgboost share torch's libomp on macOS so only one OpenMP runtime loads.
#
# Why: the pip/uv xgboost wheel links libomp via @rpath -> Homebrew's libomp,
# while torch bundles its own. Two LLVM OpenMP runtimes in one process segfault
# when both spin up thread pools (e.g. an mlp+xgboost sweep, or any run that
# imports torch and then fits xgboost). Repointing xgboost's rpath at torch's
# libomp dir leaves a single runtime in the process.
#
# Re-run this after every `uv sync` -- syncing regenerates .venv and drops the patch.
# No-op on Linux (one shared libgomp) and when already patched.
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || { echo "Not macOS; nothing to do."; exit 0; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VIRTUAL_ENV:-$REPO_ROOT/.venv}"

XGB_DYLIB="$(find "$VENV" -path '*/xgboost/lib/libxgboost.dylib' 2>/dev/null | head -1)"
TORCH_LIB_DIR="$(find "$VENV" -path '*/torch/lib' -type d 2>/dev/null | head -1)"

[[ -n "$XGB_DYLIB" ]]     || { echo "xgboost not found in $VENV; skipping."; exit 0; }
[[ -n "$TORCH_LIB_DIR" ]] || { echo "torch not found in $VENV; skipping."; exit 0; }

if otool -l "$XGB_DYLIB" | grep -q "path $TORCH_LIB_DIR"; then
  echo "Already patched -> xgboost uses torch's libomp."
  exit 0
fi

# Swap whichever libomp rpath xgboost currently carries for torch's lib dir.
OLD_RPATH="$(otool -l "$XGB_DYLIB" | awk '/LC_RPATH/{f=1} f&&/path /{print $2; exit}')"
[[ -n "$OLD_RPATH" ]] || { echo "No LC_RPATH on xgboost dylib; unexpected."; exit 1; }

install_name_tool -rpath "$OLD_RPATH" "$TORCH_LIB_DIR" "$XGB_DYLIB"
echo "Patched: xgboost rpath $OLD_RPATH -> $TORCH_LIB_DIR"
