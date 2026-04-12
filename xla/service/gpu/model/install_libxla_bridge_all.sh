#!/usr/bin/env bash
# Build //xla/service/gpu/model:libxla_bridge_all.so and copy it to each directory in
# BRIDGE_INSTALL_DIRS (below). Run from anywhere; the OpenXLA repo root is found via git.
set -euo pipefail

# Directories that should receive a copy of libxla_bridge_all.so (edit as needed).
BRIDGE_INSTALL_DIRS=(
  "/data/home/arul/dev/tunnel"
  "/data/home/arul/dev/perilune"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/WORKSPACE" ]]; then
  echo "error: could not find OpenXLA repo root (WORKSPACE) from $SCRIPT_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Building //xla/service/gpu/model:libxla_bridge_all.so in $REPO_ROOT ..."
bazel build //xla/service/gpu/model:libxla_bridge_all.so

ARTIFACT="$REPO_ROOT/bazel-bin/xla/service/gpu/model/libxla_bridge_all.so"
if [[ ! -f "$ARTIFACT" ]]; then
  echo "error: expected artifact missing: $ARTIFACT" >&2
  exit 1
fi

for dest in "${BRIDGE_INSTALL_DIRS[@]}"; do
  [[ -z "$dest" ]] && continue
  mkdir -p "$dest"
  cp -f "$ARTIFACT" "$dest/"
  echo "Installed: $dest/libxla_bridge_all.so"
done
