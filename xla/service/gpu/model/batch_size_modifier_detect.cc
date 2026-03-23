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

// Detection and parsing helpers (ported from TokenSim analyze_batch_size_dimensions).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/strings/string_view.h"
#include "absl/container/flat_hash_map.h"
#include "xla/service/gpu/model/batch_size_modifier_core.h"
#include "xla/service/gpu/model/batch_size_modifier_internal.h"

namespace xla {
namespace gpu {

void IdentifyBatchSizeInDimensions(
    const std::vector<int>& dimensions, int batch_size, int seq_len,
    int num_experts_per_tok, int num_experts, int ep,
    const std::optional<int>& num_heads,
    const std::optional<int>& tensor_count, int sp, bool* has_explicit_batch,
    int* explicit_batch, bool* batch_folded, int* folded_dim,
    std::string* pattern) {
  *has_explicit_batch = false;
  *explicit_batch = 0;
  *batch_folded = false;
  *folded_dim = -1;
  pattern->clear();

  int effective_seq_len = seq_len;
  if (sp > 1) {
    effective_seq_len = seq_len / sp;
  }

  if (batch_size > 0) {
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == batch_size) {
        *has_explicit_batch = true;
        *explicit_batch = batch_size;
        *pattern = "batch_size";
        return;
      }
    }
  }

  if (batch_size <= 0) {
    return;
  }

  auto max_padding = [](int base) {
    return std::min(512, std::max(100, base / 100));
  };

  if (seq_len > 0) {
    int expected_dim_full = batch_size * seq_len;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      int dim = dimensions[i];
      if (dim == expected_dim_full) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*seq_len";
        return;
      }
      if (dim > expected_dim_full) {
        int padding = dim - expected_dim_full;
        if (padding <= max_padding(expected_dim_full)) {
          *batch_folded = true;
          *folded_dim = static_cast<int>(i);
          *pattern = "batch_size*seq_len";
          return;
        }
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1) {
    int expected_dim_sharded = batch_size * effective_seq_len;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      int dim = dimensions[i];
      if (dim == expected_dim_sharded) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*seq_len_sharded";
        return;
      }
      if (dim > expected_dim_sharded) {
        int padding = dim - expected_dim_sharded;
        if (padding <= max_padding(expected_dim_sharded)) {
          *batch_folded = true;
          *folded_dim = static_cast<int>(i);
          *pattern = "batch_size*seq_len_sharded";
          return;
        }
        // Do not treat oversized dims as batch*seq_sharded (aligns with
        // intended analyze_batch_size_dimensions semantics; avoids false
        // positives on unrelated large leading dimensions).
        continue;
      }
    }
  }

  if (seq_len > 0 && num_experts_per_tok > 0) {
    int expected_dim_full = batch_size * seq_len * num_experts_per_tok;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim_full) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*seq_len*num_experts_per_tok";
        return;
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1 && num_experts_per_tok > 0) {
    int expected_dim_sharded =
        batch_size * effective_seq_len * num_experts_per_tok;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim_sharded) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*seq_len_sharded*num_experts_per_tok";
        return;
      }
    }
  }

  if (num_experts_per_tok > 0) {
    int expected_dim = batch_size * num_experts_per_tok;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*num_experts_per_tok";
        return;
      }
    }
  }

  if (num_heads.has_value() && *num_heads > 0) {
    int expected_dim = batch_size * (*num_heads);
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*num_heads";
        return;
      }
    }
  }

  if (num_heads.has_value() && *num_heads > 0 && seq_len > 0) {
    int expected_dim_full = batch_size * (*num_heads) * seq_len;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim_full) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*num_heads*seq_len";
        return;
      }
    }
  }

  if (num_heads.has_value() && *num_heads > 0 && effective_seq_len > 0 &&
      sp > 1) {
    int expected_dim_sharded = batch_size * (*num_heads) * effective_seq_len;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == expected_dim_sharded) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "batch_size*num_heads*seq_len_sharded";
        return;
      }
    }
  }

  if (seq_len > 0 && num_experts > 0 && ep > 0) {
    int divisor = num_experts / ep;
    if (divisor > 0) {
      int expected_dim_full = (batch_size * seq_len) / divisor;
      for (size_t i = 0; i < dimensions.size(); ++i) {
        if (std::abs(dimensions[i] - expected_dim_full) <= 1) {
          *batch_folded = true;
          *folded_dim = static_cast<int>(i);
          *pattern = "(batch_size*seq_len)/(num_experts/EP)";
          return;
        }
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1 && num_experts > 0 && ep > 0) {
    int divisor = num_experts / ep;
    if (divisor > 0) {
      int expected_dim_sharded =
          (batch_size * effective_seq_len) / divisor;
      for (size_t i = 0; i < dimensions.size(); ++i) {
        if (std::abs(dimensions[i] - expected_dim_sharded) <= 1) {
          *batch_folded = true;
          *folded_dim = static_cast<int>(i);
          *pattern = "(batch_size*seq_len_sharded)/(num_experts/EP)";
          return;
        }
      }
    }
  }

  if (seq_len > 0 && num_experts_per_tok > 0 && ep > 0) {
    int truncate_size_full = static_cast<int>(std::lround(
        (static_cast<double>(batch_size) * seq_len * num_experts_per_tok * 2) /
        ep));
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - truncate_size_full) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "round(batch_size*seq_len*num_experts_per_tok*2)/EP";
        return;
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1 && num_experts_per_tok > 0 && ep > 0) {
    int truncate_size_sharded = static_cast<int>(std::lround(
        (static_cast<double>(batch_size) * effective_seq_len *
         num_experts_per_tok * 2) /
        ep));
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - truncate_size_sharded) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "round(batch_size*seq_len_sharded*num_experts_per_tok*2)/EP";
        return;
      }
    }
  }

  if (seq_len > 0 && tensor_count.has_value() && *tensor_count > 0) {
    int expected_dim_full = (batch_size * seq_len) / (*tensor_count);
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - expected_dim_full) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "(batch_size*seq_len)/tensor_count";
        return;
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1 && tensor_count.has_value() &&
      *tensor_count > 0) {
    int expected_dim_sharded =
        (batch_size * effective_seq_len) / (*tensor_count);
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - expected_dim_sharded) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "(batch_size*seq_len_sharded)/tensor_count";
        return;
      }
    }
  }

  if (seq_len > 0 && ep > 0) {
    int expected_dim_full = (batch_size * seq_len) / ep;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - expected_dim_full) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "(batch_size*seq_len)/EP";
        return;
      }
    }
  }

  if (effective_seq_len > 0 && sp > 1 && ep > 0) {
    int expected_dim_sharded = (batch_size * effective_seq_len) / ep;
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (std::abs(dimensions[i] - expected_dim_sharded) <= 1) {
        *batch_folded = true;
        *folded_dim = static_cast<int>(i);
        *pattern = "(batch_size*seq_len_sharded)/EP";
        return;
      }
    }
  }

  if (!*batch_folded && !*has_explicit_batch && batch_size > 0) {
    for (size_t i = 0; i < dimensions.size(); ++i) {
      int dim = dimensions[i];
      if (dim > 0 && dim % batch_size == 0) {
        int multiplier = dim / batch_size;
        bool ok_heads = !num_heads.has_value() || multiplier != *num_heads;
        if (multiplier >= 2 && multiplier <= 1000 &&
            multiplier != num_experts_per_tok && ok_heads) {
          *batch_folded = true;
          *folded_dim = static_cast<int>(i);
          *pattern = absl::StrCat("batch_size*", multiplier);
          return;
        }
      }
    }
  }
}

absl::StatusOr<std::tuple<int, int, int>> InferMeshShapeFromPath(
    const std::string& file_path) {
  static const std::regex mesh_pattern(R"((\d+)x(\d+)x(\d+))");
  size_t pos = 0;
  while (pos < file_path.size()) {
    std::smatch m;
    if (std::regex_search(file_path.begin() + pos, file_path.end(), m,
                          mesh_pattern)) {
      return std::make_tuple(std::stoi(m[1].str()), std::stoi(m[2].str()),
                             std::stoi(m[3].str()));
    }
    pos++;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Cannot extract mesh_shape from path: ", file_path,
      " (expected NxMxK in a path component)"));
}

std::vector<std::pair<int, std::string>> StripMetadataSections(
    const std::vector<std::string>& lines) {
  std::vector<std::pair<int, std::string>> out;
  bool in_metadata = false;
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string& line = lines[i];
    int line_num = static_cast<int>(i + 1);
    std::string stripped(absl::StripAsciiWhitespace(line));
    if (stripped == "FileNames" || stripped == "FunctionNames" ||
        stripped == "FileLocations") {
      in_metadata = true;
      continue;
    }
    if (in_metadata) {
      if (stripped.empty()) {
        in_metadata = false;
        continue;
      }
      if (absl::StartsWith(stripped, "%") ||
          absl::StartsWith(stripped, "HloModule")) {
        in_metadata = false;
      } else {
        continue;
      }
    }
    out.push_back({line_num, line});
  }
  return out;
}

bool IsMetadataHeaderLine(absl::string_view line) {
  std::string s = std::string(absl::StripAsciiWhitespace(line));
  return s == "FileNames" || s == "FunctionNames" || s == "FileLocations";
}

std::vector<int> ParseDimensionInts(absl::string_view dims_str) {
  std::vector<int> dims;
  for (absl::string_view part :
       absl::StrSplit(dims_str, absl::ByAnyChar(", \t"), absl::SkipEmpty())) {
    part = absl::StripAsciiWhitespace(part);
    if (part.empty()) continue;
    bool all_digit = true;
    for (char c : part) {
      if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
        all_digit = false;
        break;
      }
    }
    if (all_digit) {
      dims.push_back(std::stoi(std::string(part)));
    }
  }
  return dims;
}

void BuildVariableToTensorMap(
    const std::string& content,
    absl::flat_hash_map<std::string, TensorMapEntry>* out) {
  static const std::regex pattern(R"(%([\w.]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\])");
  std::vector<std::string> lines = absl::StrSplit(content, '\n');
  for (size_t i = 0; i < lines.size(); ++i) {
    std::smatch m;
    if (std::regex_search(lines[i], m, pattern)) {
      std::vector<int> dims = ParseDimensionInts(m[3].str());
      if (!dims.empty()) {
        (*out)[m[1].str()] = TensorMapEntry{
            m[1].str(), m[2].str(), dims, static_cast<int>(i + 1)};
      }
    }
  }
}

std::optional<CommOpParsed> TryParseCommOp(const std::string& line,
                                           int line_number) {
  static const std::regex all_gather1(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+all-gather\(%([\w.\-]+))");
  static const std::regex all_gather2(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\]\s+all-gather\(%([\w.\-]+))");
  static const std::regex psum(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+psum\(%([\w.\-]+))");
  static const std::regex psum_scatter(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+psum-scatter\(%([\w.\-]+))");
  static const std::regex reduce_scatter1(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+reduce-scatter\(%([\w.\-]+))");
  static const std::regex reduce_scatter2(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\]\s+reduce-scatter\(%([\w.\-]+))");
  static const std::regex all_reduce(
      R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+all-reduce\(%([\w.\-]+))");

  std::smatch m;
  auto try_pat = [&](const std::regex& re, const char* kind)
      -> std::optional<CommOpParsed> {
    if (std::regex_search(line, m, re)) {
      CommOpParsed p;
      p.result_var = m[1].str();
      p.result_type = m[2].str();
      p.result_dims = ParseDimensionInts(m[3].str());
      p.operand_var = m[4].str();
      p.op_kind = kind;
      return p;
    }
    return std::nullopt;
  };

  if (auto x = try_pat(all_gather1, "all-gather")) return x;
  if (auto x = try_pat(all_gather2, "all-gather")) return x;
  if (auto x = try_pat(psum, "psum")) return x;
  if (auto x = try_pat(psum_scatter, "psum_scatter")) return x;
  if (auto x = try_pat(reduce_scatter1, "reduce-scatter")) return x;
  if (auto x = try_pat(reduce_scatter2, "reduce-scatter")) return x;
  if (auto x = try_pat(all_reduce, "all-reduce")) return x;
  (void)line_number;
  return std::nullopt;
}

bool LineLooksLikeCommOpRegex(const std::string& line) {
  static const std::regex assign(R"(%[\w.\-]+\s*=\s*[a-zA-Z0-9]+\[)");
  if (!std::regex_search(line, assign)) return false;
  static const std::regex patterns[] = {
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+all-gather\(%)"),
      std::regex(R"(\[[^\]]+\]\s+all-gather\(%)"),
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+psum\(%)"),
      std::regex(R"(\[[^\]]+\]\s+psum\(%)"),
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+psum_scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\s+psum_scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+psum-scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\s+psum-scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+reduce-scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\s+reduce-scatter\(%)"),
      std::regex(R"(\[[^\]]+\]\{[^}]+\}\s+all-reduce\(%)"),
      std::regex(R"(\[[^\]]+\]\s+all-reduce\(%)"),
  };
  for (const auto& p : patterns) {
    if (std::regex_search(line, p)) return true;
  }
  return false;
}

std::optional<std::string> ExtractAssignedVarName(const std::string& line) {
  static const std::regex var_pat(R"(%([\w.\-]+)\s*=)");
  std::smatch m;
  if (std::regex_search(line, m, var_pat)) {
    return m[1].str();
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
