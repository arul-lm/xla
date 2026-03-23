#!/usr/bin/env bash
# Build //xla/service/gpu/model:batch_size_modifier and run validate_batch_size_modification.py
# with --batch-size-modifier-bin and --xla-root set. Pass through all other flags to the Python script.
#
# Example:
#   ./run_validate_batch_size_modification.sh \
#     --seq-len 8192 --strategy prefill --max-batch-size 64 \
#     --dtype fp8 --quant q0 --mesh-shape 1x72x1 \
#     --config /path/to/vinveli/model_configs/deepseek_r1.yaml \
#     --hardware-arch b200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script lives at <repo>/xla/tools/batch_size_modifier/
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${REPO_ROOT}/bazel-bin/xla/service/gpu/model/batch_size_modifier"
PY="${SCRIPT_DIR}/validate_batch_size_modification.py"

cd "${REPO_ROOT}"
echo "Building batch_size_modifier in ${REPO_ROOT} ..."
bazel build //xla/service/gpu/model:batch_size_modifier

if [[ ! -f "${BIN}" ]]; then
  echo "error: expected binary not found after build: ${BIN}" >&2
  exit 1
fi

exec python3 "${PY}" "$@" \
  --batch-size-modifier-bin "${BIN}" \
  --xla-root "${REPO_ROOT}"
