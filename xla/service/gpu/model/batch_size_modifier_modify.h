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

#ifndef XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_MODIFY_H_
#define XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_MODIFY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/service/gpu/model/batch_size_modifier_core.h"
#include "xla/service/gpu/model/batch_size_modifier_internal.h"

namespace xla {
namespace gpu {

struct CsvInstructionRow {
  std::string opcode;
};

struct BackwardInstInfo {
  std::string var_name;
  std::string tensor_type;
  std::vector<int> dimensions;
  int line_num = 0;
  std::string line;
};

class BatchSizeDependencyTracker;

// Rewrites HLO-variant text (often stored with a .mlir extension) to change
// batch-related dimensions. Ported from TokenSim modify_batch_size_v4.py.
class BatchSizeModifier {
 public:
  BatchSizeModifier(
      std::string content, int old_batch_size, int new_batch_size,
      const BatchSizeModelConfig& config,
      const absl::flat_hash_map<std::string, CsvInstructionRow>& comm_csv,
      const absl::flat_hash_map<std::string, CsvInstructionRow>& comp_csv,
      bool enable_reshape_fix, bool strict_mode);

  ~BatchSizeModifier();

  absl::StatusOr<std::string> Modify();

 private:
  struct CommTensorReplace {
    std::string tensor_type;
    std::vector<int> old_dims;
    std::vector<int> new_dims;
  };

  void AnalyzeOperations();
  std::string ReplaceConstantsAndSlices(const std::string& line);
  std::string ReplaceAllTensorBracketDims(const std::string& line);
  std::vector<std::string> FixReshapeOperations(
      const std::vector<std::string>& lines);

  std::string content_;
  std::vector<std::string> lines_;
  int old_batch_size_;
  int new_batch_size_;
  BatchSizeModelConfig config_;
  absl::flat_hash_map<std::string, CsvInstructionRow> comm_csv_;
  absl::flat_hash_map<std::string, CsvInstructionRow> comp_csv_;
  bool use_comm_csv_;
  bool enable_reshape_fix_;
  bool strict_mode_;

  absl::flat_hash_map<std::string, TensorMapEntry> var_to_tensor_;
  absl::flat_hash_map<int, CommTensorReplace> comm_result_replacements_;
  absl::flat_hash_map<int, BackwardInstInfo> backward_line_map_;
  absl::flat_hash_map<int, std::vector<BackwardInstInfo>> backward_by_comm_line_;

  std::vector<std::string> undetected_msgs_;

  std::unique_ptr<BatchSizeDependencyTracker> tracker_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_MODIFY_H_
