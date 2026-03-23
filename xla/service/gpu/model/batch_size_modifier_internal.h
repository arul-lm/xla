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

#ifndef XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_INTERNAL_H_
#define XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_INTERNAL_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace gpu {

void IdentifyBatchSizeInDimensions(
    const std::vector<int>& dimensions, int batch_size, int seq_len,
    int num_experts_per_tok, int num_experts, int ep,
    const std::optional<int>& num_heads,
    const std::optional<int>& tensor_count, int sp, bool* has_explicit_batch,
    int* explicit_batch, bool* batch_folded, int* folded_dim,
    std::string* pattern);

std::vector<std::pair<int, std::string>> StripMetadataSections(
    const std::vector<std::string>& lines);

bool IsMetadataHeaderLine(absl::string_view line);

std::vector<int> ParseDimensionInts(absl::string_view dims_str);

struct TensorMapEntry {
  std::string var_name;
  std::string tensor_type;
  std::vector<int> dimensions;
  int line_num = 0;
};

void BuildVariableToTensorMap(
    const std::string& content,
    absl::flat_hash_map<std::string, TensorMapEntry>* out);

struct CommOpParsed {
  std::string result_var;
  std::string result_type;
  std::vector<int> result_dims;
  std::string operand_var;
  std::string op_kind;
};

std::optional<CommOpParsed> TryParseCommOp(const std::string& line,
                                             int line_number);

bool LineLooksLikeCommOpRegex(const std::string& line);

std::optional<std::string> ExtractAssignedVarName(const std::string& line);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_INTERNAL_H_
