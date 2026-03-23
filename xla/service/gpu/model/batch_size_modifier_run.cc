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

#include "xla/service/gpu/model/batch_size_modifier_core.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/strip.h"
#include "absl/strings/str_cat.h"
#include "xla/service/gpu/model/batch_size_modifier_internal.h"
#include "xla/service/gpu/model/batch_size_modifier_modify.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

absl::Status ReadFileUtf8(const std::string& path, std::string* out) {
  return tsl::ReadFileToString(tsl::Env::Default(), path, out);
}

absl::Status WriteFileUtf8(const std::string& path, const std::string& data) {
  return tsl::WriteStringToFile(tsl::Env::Default(), path, data);
}

// Minimal parser for vinveli YAML with a top-level `modify_batch_size:` block.
absl::StatusOr<BatchSizeModelConfig> ParseModifyBatchSizeSection(
    const std::string& yaml_text) {
  BatchSizeModelConfig cfg;
  std::istringstream in(yaml_text);
  std::string line;
  bool in_section = false;
  int base_indent = -1;
  while (std::getline(in, line)) {
    std::string raw = line;
    std::string trimmed = std::string(absl::StripAsciiWhitespace(line));
    if (trimmed.empty() || trimmed[0] == '#') continue;
    if (!in_section) {
      if (trimmed == "modify_batch_size:" ||
        absl::StartsWith(trimmed, "modify_batch_size:")) {
        in_section = true;
        base_indent = -1;
      }
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(line[0])) &&
        trimmed.find(':') != std::string::npos &&
        trimmed.find("modify_batch_size") == std::string::npos) {
      // New top-level key — end section.
      break;
    }
    size_t first_non_space = 0;
    while (first_non_space < line.size() &&
           std::isspace(static_cast<unsigned char>(line[first_non_space]))) {
      ++first_non_space;
    }
    if (base_indent < 0) {
      base_indent = static_cast<int>(first_non_space);
    }
    if (static_cast<int>(first_non_space) < base_indent) {
      break;
    }
    size_t colon = trimmed.find(':');
    if (colon == std::string::npos) continue;
    std::string key = std::string(absl::StripAsciiWhitespace(trimmed.substr(0, colon)));
    std::string val = std::string(absl::StripAsciiWhitespace(trimmed.substr(colon + 1)));
    if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
      val = val.substr(1, val.size() - 2);
    }
    if (key == "num_experts") {
      cfg.num_experts = std::stoi(val);
    } else if (key == "seq_len") {
      cfg.seq_len = std::stoi(val);
    } else if (key == "num_experts_per_tok") {
      cfg.num_experts_per_tok = std::stoi(val);
    } else if (key == "sp") {
      cfg.sp = std::stoi(val);
    } else if (key == "num_heads") {
      int v = std::stoi(val);
      if (v > 0) cfg.num_heads = v;
    }
  }
  if (cfg.num_experts <= 0 || cfg.seq_len <= 0 || cfg.num_experts_per_tok <= 0) {
    return absl::InvalidArgumentError(
        "modify_batch_size section missing required fields "
        "(num_experts, seq_len, num_experts_per_tok)");
  }
  if (cfg.sp <= 0) {
    return absl::InvalidArgumentError("modify_batch_size.sp is required");
  }
  return cfg;
}

absl::StatusOr<absl::flat_hash_map<std::string, CsvInstructionRow>>
LoadInstructionCsv(const std::string& path) {
  absl::flat_hash_map<std::string, CsvInstructionRow> out;
  std::ifstream f(path);
  if (!f) {
    return absl::InvalidArgumentError(absl::StrCat("Cannot open CSV: ", path));
  }
  std::string header_line;
  if (!std::getline(f, header_line)) {
    return out;
  }
  auto split_csv = [](const std::string& row) {
    std::vector<std::string> cells;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < row.size(); ++i) {
      char c = row[i];
      if (c == '"') {
        in_quotes = !in_quotes;
      } else if (c == ',' && !in_quotes) {
        cells.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
    cells.push_back(cur);
    return cells;
  };
  std::vector<std::string> headers = split_csv(header_line);
  int idx_name = -1;
  int idx_opcode = -1;
  for (size_t i = 0; i < headers.size(); ++i) {
    std::string h = std::string(absl::StripAsciiWhitespace(headers[i]));
    if (h == "instruction_name") idx_name = static_cast<int>(i);
    if (h == "opcode") idx_opcode = static_cast<int>(i);
  }
  if (idx_name < 0 || idx_opcode < 0) {
    return absl::InvalidArgumentError(
        "CSV must contain instruction_name and opcode columns");
  }
  std::string row;
  while (std::getline(f, row)) {
    std::vector<std::string> cells = split_csv(row);
    if (static_cast<int>(cells.size()) <= std::max(idx_name, idx_opcode)) {
      continue;
    }
    std::string name =
        std::string(absl::StripAsciiWhitespace(cells[idx_name]));
    std::string opcode =
        std::string(absl::StripAsciiWhitespace(cells[idx_opcode]));
    if (name.empty()) continue;
    out[name] = CsvInstructionRow{opcode};
  }
  return out;
}

}  // namespace

absl::StatusOr<BatchSizeModelConfig> LoadBatchSizeModelConfigFromYaml(
    const std::string& config_path, int ep_from_path) {
  std::string text;
  TF_RETURN_IF_ERROR(ReadFileUtf8(config_path, &text));
  absl::StatusOr<BatchSizeModelConfig> cfg_or =
      ParseModifyBatchSizeSection(text);
  if (!cfg_or.ok()) return cfg_or.status();
  BatchSizeModelConfig cfg = *std::move(cfg_or);
  cfg.ep = ep_from_path;
  return cfg;
}

absl::Status RunBatchSizeModification(const BatchSizeModifierOptions& opts) {
  if (opts.old_batch_size <= 0 || opts.new_batch_size <= 0) {
    return absl::InvalidArgumentError("batch sizes must be positive");
  }
  if (opts.old_batch_size == opts.new_batch_size) {
    std::string data;
    TF_RETURN_IF_ERROR(ReadFileUtf8(opts.input_mlir_path, &data));
    return WriteFileUtf8(opts.output_mlir_path, data);
  }
  std::string mesh_path = opts.path_for_mesh_inference.empty()
                              ? opts.input_mlir_path
                              : opts.path_for_mesh_inference;
  auto mesh = InferMeshShapeFromPath(mesh_path);
  if (!mesh.ok()) return mesh.status();
  int sp_axis = std::get<0>(*mesh);
  int ep_axis = std::get<1>(*mesh);
  int tensor_axis = std::get<2>(*mesh);

  absl::StatusOr<BatchSizeModelConfig> cfg_or =
      LoadBatchSizeModelConfigFromYaml(opts.config_yaml_path, ep_axis);
  if (!cfg_or.ok()) return cfg_or.status();
  BatchSizeModelConfig cfg = *std::move(cfg_or);
  cfg.tensor_count = tensor_axis;

  absl::flat_hash_map<std::string, CsvInstructionRow> comm_csv;
  absl::flat_hash_map<std::string, CsvInstructionRow> comp_csv;
  if (!opts.comm_stats_csv_path.empty()) {
    auto c = LoadInstructionCsv(opts.comm_stats_csv_path);
    if (!c.ok()) return c.status();
    comm_csv = std::move(*c);
  }
  if (!opts.comp_stats_csv_path.empty()) {
    auto c = LoadInstructionCsv(opts.comp_stats_csv_path);
    if (!c.ok()) return c.status();
    comp_csv = std::move(*c);
  }

  std::string content;
  TF_RETURN_IF_ERROR(ReadFileUtf8(opts.input_mlir_path, &content));

  BatchSizeModifier modifier(content, opts.old_batch_size, opts.new_batch_size,
                             cfg, comm_csv, comp_csv, opts.enable_reshape_fix,
                             opts.strict_mode);
  auto modified = modifier.Modify();
  if (!modified.ok()) return modified.status();
  return WriteFileUtf8(opts.output_mlir_path, *modified);
}

}  // namespace gpu
}  // namespace xla
