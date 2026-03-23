#!/usr/bin/env bash
#
# Build //xla/service/gpu/model:batch_size_modifier, then run validate_batch_size_modification.py.
#
# This script always appends (after your arguments):
#   --batch-size-modifier-bin <bazel-bin path>
#   --xla-root <OpenXLA repo root>
# so you do not need BATCH_SIZE_MODIFIER_BIN or XLA_ROOT. If you also pass those flags,
# the appended values usually win (last occurrence).
#
# All other arguments are forwarded to validate_batch_size_modification.py. Required there:
#
#   --seq-len INT              Sequence length
#   --strategy {prefill|decode}
#   --max-batch-size INT       Reference batch (old_batch_size for the modifier); sweep uses [1, max]
#   --dtype STR                e.g. fp8, bf16
#   --quant STR                e.g. q0, q1
#   --mesh-shape STR           e.g. 1x72x1  (SPxEPxTP in path / config)
#   --config PATH              Model YAML (e.g. vinveli model_configs/deepseek_r1.yaml)
#   --hardware-arch STR        e.g. b200 or b200,b200l200 (analytical calculator)
#
# Optional:
#
#   --num-datapoints INT       Batch sizes to sample in [1, max]; default 10
#   --output-dir PATH          Outputs; default temp dir
#   --vinveli-home PATH        Default: VINVELI_HOME env
#   --overlap-factor FLOAT     Default 0.5
#   --max-workers INT          Passed to run_sdy_generator_batch only; default 1
#   --skip-cleanup             Do not delete existing hlo_* dirs before generation
#   --xla-container-path STR   Default /xla/hlo (docker copy target)
#   --container-name STR       Default xla
#
# Environment (if flags omitted):
#   VINVELI_HOME               Required unless --vinveli-home is set
#
# Example:
#   ./run_validate_batch_size_modification.sh \
#     --seq-len 8192 --strategy prefill --max-batch-size 64 \
#     --dtype fp8 --quant q0 --mesh-shape 1x72x1 \
#     --config /path/to/vinveli/model_configs/deepseek_r1.yaml \
#     --hardware-arch b200 \
#     --num-datapoints 8 --max-workers 4

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
