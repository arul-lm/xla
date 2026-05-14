#!/usr/bin/env bash
# Build //xla/service/gpu/model:libxla_bridge_all.so inside the Docker
# container and copy the resulting artifact to each directory in
# BRIDGE_INSTALL_DIRS (below). Run from anywhere; the OpenXLA repo root is
# found via git.
#
# Why Docker: the host toolchain doesn't match Bazel's pinned toolchain, and
# bazel-bin/ is a symlink into the container's /root/.cache that the host
# can't read directly. So we build in-container and use `docker cp` to
# extract the .so before fanning it out to the install dirs.
set -euo pipefail

# Directories that should receive a copy of libxla_bridge_all.so (edit as needed).
# Each path is the directory the Rust crate's build.rs links against
# (cargo:rustc-link-search=native=<here>); see e.g. tunnel/build.rs and
# perilune/build.rs which both point at <crate>/xla_libs.
BRIDGE_INSTALL_DIRS=(
  "/data/home/arul/dev/tunnel/xla_libs"
  "/data/home/arul/dev/perilune/xla_libs"
)

# Docker container that has the Bazel + XLA toolchain. Override via env.
DOCKER_CONTAINER="${DOCKER_CONTAINER:-xla}"
# Path to the XLA repo root inside the container.
DOCKER_REPO_ROOT="${DOCKER_REPO_ROOT:-/xla}"

BAZEL_TARGET="//xla/service/gpu/model:libxla_bridge_all.so"
ARTIFACT_REL="bazel-bin/xla/service/gpu/model/libxla_bridge_all.so"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/WORKSPACE" ]]; then
  echo "error: could not find OpenXLA repo root (WORKSPACE) from $SCRIPT_DIR" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: 'docker' not found on PATH" >&2
  exit 1
fi
if ! docker inspect "$DOCKER_CONTAINER" >/dev/null 2>&1; then
  echo "error: docker container '$DOCKER_CONTAINER' not found (override with DOCKER_CONTAINER=...)" >&2
  exit 1
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$DOCKER_CONTAINER")" != "true" ]]; then
  echo "error: docker container '$DOCKER_CONTAINER' is not running" >&2
  exit 1
fi

echo "Building $BAZEL_TARGET inside container '$DOCKER_CONTAINER' ..."
docker exec "$DOCKER_CONTAINER" bash -c "cd '$DOCKER_REPO_ROOT' && bazel build $BAZEL_TARGET"

TMP_ARTIFACT="$(mktemp -t libxla_bridge_all.XXXXXX.so)"
trap 'rm -f "$TMP_ARTIFACT"' EXIT

echo "Extracting artifact via docker cp ..."
docker cp "$DOCKER_CONTAINER:$DOCKER_REPO_ROOT/$ARTIFACT_REL" "$TMP_ARTIFACT"
if [[ ! -f "$TMP_ARTIFACT" ]]; then
  echo "error: docker cp produced no file at $TMP_ARTIFACT" >&2
  exit 1
fi

ARTIFACT_SHA="$(sha256sum "$TMP_ARTIFACT" | awk '{print $1}')"
ARTIFACT_SIZE="$(stat -c '%s' "$TMP_ARTIFACT")"
echo "Artifact: $ARTIFACT_SIZE bytes  sha256=$ARTIFACT_SHA"

for dest in "${BRIDGE_INSTALL_DIRS[@]}"; do
  [[ -z "$dest" ]] && continue
  mkdir -p "$dest"
  cp -f "$TMP_ARTIFACT" "$dest/libxla_bridge_all.so"
  chmod +rx "$dest/libxla_bridge_all.so"
  echo "Installed: $dest/libxla_bridge_all.so"
done

echo "Done. Reminder: any process that has the old .so mmapped (e.g. a running"
echo "tunnel/perilune server) must be restarted to pick up the new build."
