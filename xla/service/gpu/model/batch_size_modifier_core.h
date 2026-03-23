/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_CORE_H_
#define XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_CORE_H_

#include <optional>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xla {
namespace gpu {

// Model parameters (vinveli `modify_batch_size` section + mesh-derived fields).
struct BatchSizeModelConfig {
  int num_experts = 0;
  int ep = 0;
  int seq_len = 0;
  int num_experts_per_tok = 0;
  std::optional<int> num_heads;
  std::optional<int> tensor_count;
  int sp = 1;
};

struct BatchSizeModifierOptions {
  std::string input_mlir_path;
  std::string output_mlir_path;
  int old_batch_size = 0;
  int new_batch_size = 0;
  // YAML with top-level `modify_batch_size:` (vinveli config).
  std::string config_yaml_path;
  // Used to infer SP/EP/tensor_count from directory names (e.g. ..._1x72x1/...).
  // If empty, uses input_mlir_path.
  std::string path_for_mesh_inference;
  std::string comm_stats_csv_path;
  std::string comp_stats_csv_path;
  bool enable_reshape_fix = true;
  bool strict_mode = false;
};

// Loads mesh axes (SP, EP, tensor_count) from a path containing NxMxK.
absl::StatusOr<std::tuple<int, int, int>> InferMeshShapeFromPath(
    const std::string& file_path);

absl::StatusOr<BatchSizeModelConfig> LoadBatchSizeModelConfigFromYaml(
    const std::string& config_path, int ep_from_path);

// Reads input MLIR, applies batch-size rewrite, writes output. Copies input to
// output unchanged when old_batch_size == new_batch_size.
absl::Status RunBatchSizeModification(const BatchSizeModifierOptions& opts);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_CORE_H_
