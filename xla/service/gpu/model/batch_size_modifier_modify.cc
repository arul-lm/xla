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

#include "xla/service/gpu/model/batch_size_modifier_modify.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <regex>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/model/batch_size_modifier_internal.h"

namespace xla {
namespace gpu {
namespace {

struct Detection {
  bool has_explicit_batch = false;
  bool has_folded_batch = false;
  int explicit_batch_size = 0;
  int folded_dimension = -1;
  std::string pattern;
};

Detection DetectDimensions(const BatchSizeModelConfig& cfg, int old_batch,
                           const std::vector<int>& dimensions) {
  Detection d;
  IdentifyBatchSizeInDimensions(
      dimensions, old_batch, cfg.seq_len, cfg.num_experts_per_tok,
      cfg.num_experts, cfg.ep, cfg.num_heads, cfg.tensor_count, cfg.sp,
      &d.has_explicit_batch, &d.explicit_batch_size, &d.has_folded_batch,
      &d.folded_dimension, &d.pattern);
  return d;
}

std::optional<int> CalculateNewDimensionFromPattern(
    const BatchSizeModelConfig& cfg, int new_batch,
    const std::string& pattern, int old_dim, int old_batch) {
  int effective_seq_len = cfg.seq_len;
  if (cfg.sp > 1) {
    effective_seq_len = cfg.seq_len / cfg.sp;
  }
  if (pattern == "batch_size") {
    return new_batch;
  }
  if (pattern == "batch_size*seq_len") {
    return new_batch * cfg.seq_len;
  }
  if (pattern == "batch_size*seq_len_sharded") {
    return new_batch * effective_seq_len;
  }
  if (pattern == "batch_size*seq_len*num_experts_per_tok") {
    return new_batch * cfg.seq_len * cfg.num_experts_per_tok;
  }
  if (pattern == "batch_size*seq_len_sharded*num_experts_per_tok") {
    return new_batch * effective_seq_len * cfg.num_experts_per_tok;
  }
  if (pattern == "batch_size*num_experts_per_tok") {
    if (cfg.num_experts_per_tok > 0) {
      return new_batch * cfg.num_experts_per_tok;
    }
  }
  if (pattern == "batch_size*num_heads") {
    if (cfg.num_heads.has_value() && *cfg.num_heads > 0) {
      return new_batch * (*cfg.num_heads);
    }
  }
  if (pattern == "batch_size*num_heads*seq_len") {
    if (cfg.num_heads.has_value() && *cfg.num_heads > 0) {
      return new_batch * (*cfg.num_heads) * cfg.seq_len;
    }
  }
  if (pattern == "batch_size*num_heads*seq_len_sharded") {
    if (cfg.num_heads.has_value() && *cfg.num_heads > 0) {
      return new_batch * (*cfg.num_heads) * effective_seq_len;
    }
  }
  if (pattern == "(batch_size*seq_len)/(num_experts/EP)") {
    if (cfg.ep > 0) {
      int divisor = cfg.num_experts / cfg.ep;
      if (divisor > 0) {
        return (new_batch * cfg.seq_len) / divisor;
      }
    }
  }
  if (pattern == "(batch_size*seq_len_sharded)/(num_experts/EP)") {
    if (cfg.ep > 0) {
      int divisor = cfg.num_experts / cfg.ep;
      if (divisor > 0) {
        return (new_batch * effective_seq_len) / divisor;
      }
    }
  }
  if (pattern == "round(batch_size*seq_len*num_experts_per_tok*2)/EP") {
    if (cfg.ep > 0) {
      return static_cast<int>(std::lround(
          (static_cast<double>(new_batch) * cfg.seq_len *
           cfg.num_experts_per_tok * 2) /
          cfg.ep));
    }
  }
  if (pattern == "round(batch_size*seq_len_sharded*num_experts_per_tok*2)/EP") {
    if (cfg.ep > 0) {
      return static_cast<int>(std::lround(
          (static_cast<double>(new_batch) * effective_seq_len *
           cfg.num_experts_per_tok * 2) /
          cfg.ep));
    }
  }
  if (pattern == "(batch_size*seq_len)/tensor_count") {
    if (cfg.tensor_count.has_value() && *cfg.tensor_count > 0) {
      return (new_batch * cfg.seq_len) / (*cfg.tensor_count);
    }
  }
  if (pattern == "(batch_size*seq_len_sharded)/tensor_count") {
    if (cfg.tensor_count.has_value() && *cfg.tensor_count > 0) {
      return (new_batch * effective_seq_len) / (*cfg.tensor_count);
    }
  }
  if (pattern == "(batch_size*seq_len)/EP") {
    if (cfg.ep > 0) {
      return (new_batch * cfg.seq_len) / cfg.ep;
    }
  }
  if (pattern == "(batch_size*seq_len_sharded)/EP") {
    if (cfg.ep > 0) {
      return (new_batch * effective_seq_len) / cfg.ep;
    }
  }
  if (!pattern.empty() && pattern.size() > 12 &&
      pattern.substr(0, 12) == "batch_size*") {
    std::string rest = pattern.substr(12);
    char* end = nullptr;
    long mult = std::strtol(rest.c_str(), &end, 10);
    if (end != rest.c_str() && *end == '\0' && mult > 0) {
      return new_batch * static_cast<int>(mult);
    }
  }
  (void)old_dim;
  (void)old_batch;
  return std::nullopt;
}

std::optional<std::vector<int>> CalculateNewDimensions(
    const BatchSizeModelConfig& cfg, int old_batch, int new_batch,
    const std::vector<int>& dimensions, const Detection& det) {
  if (!det.has_explicit_batch && !det.has_folded_batch) {
    return std::nullopt;
  }
  std::vector<int> new_dims = dimensions;
  bool modified = false;
  if (det.has_explicit_batch) {
    for (size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == old_batch) {
        new_dims[i] = new_batch;
        modified = true;
        break;
      }
    }
  } else if (det.has_folded_batch && det.folded_dimension >= 0 &&
             static_cast<size_t>(det.folded_dimension) < dimensions.size()) {
    int idx = det.folded_dimension;
    auto nd = CalculateNewDimensionFromPattern(cfg, new_batch, det.pattern,
                                               dimensions[idx], old_batch);
    if (nd.has_value()) {
      new_dims[idx] = *nd;
      modified = true;
    }
  }
  if (!modified) return std::nullopt;
  return new_dims;
}

std::string ReplaceTensorDimsInLine(const std::string& line,
                                    const std::string& tensor_type,
                                    const std::vector<int>& old_dims,
                                    const std::vector<int>& new_dims) {
  std::string old_str = absl::StrJoin(old_dims, ",");
  std::string new_str = absl::StrJoin(new_dims, ",");
  std::regex re(absl::StrCat("(", tensor_type, R"(\[)", old_str, R"(\]))"));
  return std::regex_replace(line, re, tensor_type + "[" + new_str + "]",
                            std::regex_constants::format_first_only);
}

}  // namespace

class BatchSizeDependencyTracker {
 public:
  explicit BatchSizeDependencyTracker(
      const std::vector<std::string>& lines,
      const absl::flat_hash_map<std::string, CsvInstructionRow>& comp_csv) {
    static const std::regex def_pat(
        R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\])");
    auto filtered = StripMetadataSections(lines);
    for (const auto& pr : filtered) {
      int line_num = pr.first;
      const std::string& line = pr.second;
      std::smatch m;
      if (std::regex_search(line, m, def_pat)) {
        std::vector<int> dims = ParseDimensionInts(m[3].str());
        if (!dims.empty()) {
          var_to_instruction_[m[1].str()] = BackwardInstInfo{
              m[1].str(), m[2].str(), dims, line_num, line};
        }
      }
    }
    (void)comp_csv;
  }

  std::vector<BackwardInstInfo> TraceBackward(const std::string& operand_var) {
    absl::flat_hash_set<std::string> visited;
    return TraceBackwardRec(operand_var, &visited);
  }

 private:
  std::vector<BackwardInstInfo> TraceBackwardRec(
      const std::string& operand_var,
      absl::flat_hash_set<std::string>* visited) {
    if (visited->size() > 1000) return {};
    if (visited->contains(operand_var)) return {};
    auto it = var_to_instruction_.find(operand_var);
    if (it == var_to_instruction_.end()) return {};
    visited->insert(operand_var);
    std::vector<BackwardInstInfo> result = {it->second};
    static const std::regex op_pat(R"(%([\w.]+))");
    std::string line = it->second.line;
    std::sregex_iterator iter(line.begin(), line.end(), op_pat);
    std::sregex_iterator end;
    for (; iter != end; ++iter) {
      std::string op = (*iter)[1].str();
      if (op != operand_var) {
        auto sub = TraceBackwardRec(op, visited);
        result.insert(result.end(), sub.begin(), sub.end());
      }
    }
    return result;
  }

  absl::flat_hash_map<std::string, BackwardInstInfo> var_to_instruction_;
};

namespace {

bool IsCommLineWithCsv(
    const std::string& line,
    const absl::flat_hash_map<std::string, CsvInstructionRow>& comm_csv) {
  auto v = ExtractAssignedVarName(line);
  return v.has_value() && comm_csv.contains(*v);
}

std::optional<CommOpParsed> ParseCommLineWithCsv(
    const std::string& line, int line_num,
    const absl::flat_hash_map<std::string, CsvInstructionRow>& comm_csv) {
  auto v = ExtractAssignedVarName(line);
  if (!v.has_value() || !comm_csv.contains(*v)) return std::nullopt;
  const std::string& opcode = comm_csv.at(*v).opcode;
  if (opcode == "all-gather") {
    return TryParseCommOp(line, line_num);
  }
  if (opcode == "all-reduce") {
    std::string lower = *v;
    for (char& c : lower) c = static_cast<char>(std::tolower(c));
    if (lower.find("psum") != std::string::npos) {
      static const std::regex psum_re(
          R"(%([\w.\-]+)\s*=\s*([a-zA-Z0-9]+)\[([^\]]+)\](?:\{[^}]+\})?\s+psum\(%([\w.\-]+))");
      std::smatch m;
      if (std::regex_search(line, m, psum_re)) {
        CommOpParsed p;
        p.result_var = m[1].str();
        p.result_type = m[2].str();
        p.result_dims = ParseDimensionInts(m[3].str());
        p.operand_var = m[4].str();
        p.op_kind = "psum";
        return p;
      }
    }
    return TryParseCommOp(line, line_num);
  }
  if (opcode == "reduce-scatter") {
    return TryParseCommOp(line, line_num);
  }
  return TryParseCommOp(line, line_num);
}

int64_t Product(const std::vector<int>& dims) {
  int64_t p = 1;
  for (int d : dims) p *= d;
  return p;
}

}  // namespace

BatchSizeModifier::BatchSizeModifier(
    std::string content, int old_batch_size, int new_batch_size,
    const BatchSizeModelConfig& config,
    const absl::flat_hash_map<std::string, CsvInstructionRow>& comm_csv,
    const absl::flat_hash_map<std::string, CsvInstructionRow>& comp_csv,
    bool enable_reshape_fix, bool strict_mode)
    : content_(std::move(content)),
      old_batch_size_(old_batch_size),
      new_batch_size_(new_batch_size),
      config_(config),
      comm_csv_(comm_csv),
      comp_csv_(comp_csv),
      use_comm_csv_(!comm_csv.empty()),
      enable_reshape_fix_(enable_reshape_fix),
      strict_mode_(strict_mode) {
  lines_ = absl::StrSplit(content_, '\n');
  BuildVariableToTensorMap(content_, &var_to_tensor_);
  tracker_ = std::make_unique<BatchSizeDependencyTracker>(lines_, comp_csv_);
  AnalyzeOperations();
}

BatchSizeModifier::~BatchSizeModifier() = default;

void BatchSizeModifier::AnalyzeOperations() {
  auto filtered_lines = StripMetadataSections(lines_);

  for (const auto& pr : filtered_lines) {
    int line_num = pr.first;
    const std::string& line = pr.second;
    bool is_comm = false;
    if (use_comm_csv_) {
      is_comm = IsCommLineWithCsv(line, comm_csv_);
    } else {
      is_comm = LineLooksLikeCommOpRegex(line);
    }
    if (!is_comm) continue;

    std::optional<CommOpParsed> parsed;
    if (use_comm_csv_) {
      parsed = ParseCommLineWithCsv(line, line_num, comm_csv_);
    } else {
      if (line.find("reduce-scatter") != std::string::npos) {
        parsed = TryParseCommOp(line, line_num);
      } else if (line.find("all-gather") != std::string::npos) {
        parsed = TryParseCommOp(line, line_num);
      } else if (line.find("psum_scatter") != std::string::npos ||
                 line.find("psum-scatter") != std::string::npos) {
        parsed = TryParseCommOp(line, line_num);
      } else if (line.find("psum") != std::string::npos &&
                 line.find("psum_scatter") == std::string::npos) {
        parsed = TryParseCommOp(line, line_num);
      } else if (line.find("all-reduce") != std::string::npos) {
        parsed = TryParseCommOp(line, line_num);
      }
    }
    if (!parsed.has_value()) continue;

    CommOpParsed op = *parsed;
    if (var_to_tensor_.contains(op.operand_var)) {
      // Operand dims filled from map when needed in detection below.
    }

    absl::flat_hash_map<int, BackwardInstInfo> unique_backward;
    if (!op.result_dims.empty()) {
      Detection rd = DetectDimensions(config_, old_batch_size_, op.result_dims);
      if (!rd.has_explicit_batch && !rd.has_folded_batch) {
        undetected_msgs_.push_back(absl::StrCat(
            "Communication op ", op.result_var, " (line ", line_num,
            ") result tensor not detected"));
      } else {
        auto new_dims =
            CalculateNewDimensions(config_, old_batch_size_, new_batch_size_,
                                   op.result_dims, rd);
        if (new_dims.has_value()) {
          comm_result_replacements_[line_num] = {
              op.result_type, op.result_dims, *new_dims};
        }
      }
    }

    auto fill_operand_dims = [&](const std::string& name) {
      if (var_to_tensor_.contains(name)) {
        return var_to_tensor_.at(name).dimensions;
      }
      return std::vector<int>{};
    };

    std::string operand_name = op.operand_var;
    std::vector<int> odims = fill_operand_dims(operand_name);
    if (!odims.empty()) {
      Detection od =
          DetectDimensions(config_, old_batch_size_, odims);
      if (od.has_explicit_batch || od.has_folded_batch) {
        auto traced = tracker_->TraceBackward(operand_name);
        for (const BackwardInstInfo& inst : traced) {
          if (inst.line_num == line_num) continue;
          Detection id =
              DetectDimensions(config_, old_batch_size_, inst.dimensions);
          if (id.has_explicit_batch || id.has_folded_batch) {
            unique_backward[inst.line_num] = inst;
          }
        }
      } else {
        undetected_msgs_.push_back(absl::StrCat(
            "Operand ", operand_name, " of ", op.op_kind, " dimensions not ",
            "detected"));
      }
    }

    for (const auto& e : unique_backward) {
      backward_by_comm_line_[line_num].push_back(e.second);
    }
  }

  // Compute instructions from CSV
  if (!comp_csv_.empty()) {
    absl::flat_hash_map<std::string, std::pair<int, std::string>> var_to_line;
    for (const auto& pr : filtered_lines) {
      auto vn = ExtractAssignedVarName(pr.second);
      if (vn.has_value() && !var_to_line.contains(*vn)) {
        var_to_line[*vn] = {pr.first, pr.second};
      }
    }
    for (const auto& e : comp_csv_) {
      const std::string& inst_name = e.first;
      if (!var_to_line.contains(inst_name)) continue;
      int ln = var_to_line[inst_name].first;
      const std::string& lnstr = var_to_line[inst_name].second;
      static const std::regex tensor_bracket(
          R"(([a-zA-Z0-9]+)\[([^\]]+)\])");
      std::smatch m;
      if (std::regex_search(lnstr, m, tensor_bracket)) {
        std::vector<int> dims = ParseDimensionInts(m[2].str());
        if (!dims.empty()) {
          Detection d = DetectDimensions(config_, old_batch_size_, dims);
          if (d.has_explicit_batch || d.has_folded_batch) {
            if (!backward_line_map_.contains(ln)) {
              backward_line_map_[ln] = BackwardInstInfo{
                  inst_name, m[1].str(), dims, ln, lnstr};
            }
          }
        }
      }
    }
  }

  for (const auto& pr : backward_by_comm_line_) {
    for (const BackwardInstInfo& ii : pr.second) {
      if (!backward_line_map_.contains(ii.line_num)) {
        backward_line_map_[ii.line_num] = ii;
      }
    }
  }
}

std::string BatchSizeModifier::ReplaceConstantsAndSlices(const std::string& line) {
  std::string out = line;
  {
    std::regex c1(absl::StrCat(R"((\w+) = s32\[\] constant\()",
                               old_batch_size_, R"(\))"));
    out = std::regex_replace(
        out, c1,
        absl::StrCat("$1 = s32[] constant(", new_batch_size_, ")"));
  }
  {
    std::regex c2(absl::StrCat(R"(dimensions=\{)", old_batch_size_, R"(\})"));
    out = std::regex_replace(
        out, c2, absl::StrCat("dimensions={", new_batch_size_, "}"));
  }
  {
    std::regex c3(absl::StrCat(R"(shape=\{)", old_batch_size_, ","));
    out = std::regex_replace(out, c3,
                             absl::StrCat("shape={", new_batch_size_, ","));
  }
  if (out.find("slice=") != std::string::npos &&
      out.find("slice={") != std::string::npos) {
    static const std::regex slice_re(R"(\[(\d+):(\d+)(?::(\d+))?\])");
    struct SliceRep {
      size_t pos;
      size_t len;
      std::string text;
    };
    std::vector<SliceRep> reps;
    for (std::sregex_iterator it(out.begin(), out.end(), slice_re), end;
         it != end; ++it) {
      const std::smatch& m = *it;
      std::string start = m[1].str();
      int endv = std::stoi(m[2].str());
      bool has_step = m[3].matched;
      std::string step = has_step ? m[3].str() : "";
      std::optional<std::string> new_end;
      if (endv == old_batch_size_) {
        new_end = absl::StrCat(new_batch_size_);
      } else if (endv == old_batch_size_ * config_.seq_len) {
        new_end = absl::StrCat(new_batch_size_ * config_.seq_len);
      } else if (config_.ep > 0) {
        int divisor = config_.num_experts / config_.ep;
        if (divisor > 0) {
          int expected_routing =
              (old_batch_size_ * config_.seq_len) / divisor;
          if (std::abs(endv - expected_routing) <= 1) {
            new_end = absl::StrCat((new_batch_size_ * config_.seq_len) / divisor);
          }
        }
      }
      if (!new_end.has_value()) continue;
      std::string repl;
      if (has_step) {
        repl = absl::StrCat("[", start, ":", *new_end, ":", step, "]");
      } else {
        repl = absl::StrCat("[", start, ":", *new_end, "]");
      }
      reps.push_back({static_cast<size_t>(m.position(0)),
                      static_cast<size_t>(m.length(0)), repl});
    }
    std::sort(reps.begin(), reps.end(),
              [](const SliceRep& a, const SliceRep& b) { return a.pos > b.pos; });
    for (const SliceRep& r : reps) {
      out.replace(r.pos, r.len, r.text);
    }
  }

  static const std::regex dyn_re(R"(dynamic_slice_sizes=\{([^}]+)\})");
  struct DynRep {
    size_t pos;
    size_t len;
    std::string text;
  };
  std::vector<DynRep> dyn_reps;
  for (std::sregex_iterator it(out.begin(), out.end(), dyn_re), end;
       it != end; ++it) {
    const std::smatch& m = *it;
    std::vector<int> dims = ParseDimensionInts(m[1].str());
    if (dims.empty()) continue;
    std::vector<int> new_dims;
    bool modified = false;
    for (int dim : dims) {
      Detection det = DetectDimensions(config_, old_batch_size_, {dim});
      auto nd = CalculateNewDimensions(config_, old_batch_size_,
                                         new_batch_size_, {dim}, det);
      if (nd.has_value() && (*nd)[0] != dim) {
        new_dims.push_back((*nd)[0]);
        modified = true;
      } else {
        new_dims.push_back(dim);
      }
    }
    if (!modified) continue;
    dyn_reps.push_back(
        {static_cast<size_t>(m.position(0)), static_cast<size_t>(m.length(0)),
         absl::StrCat("dynamic_slice_sizes={", absl::StrJoin(new_dims, ","),
                      "}")});
  }
  std::sort(dyn_reps.begin(), dyn_reps.end(),
            [](const DynRep& a, const DynRep& b) { return a.pos > b.pos; });
  for (const DynRep& r : dyn_reps) {
    out.replace(r.pos, r.len, r.text);
  }

  return out;
}

std::string BatchSizeModifier::ReplaceAllTensorBracketDims(
    const std::string& line) {
  static const std::regex tensor_re(R"(([a-zA-Z0-9]+)\[([^\]]+)\])");
  struct Rep {
    size_t pos;
    size_t len;
    std::string text;
  };
  std::vector<Rep> reps;
  for (std::sregex_iterator it(line.begin(), line.end(), tensor_re), end;
       it != end; ++it) {
    const std::smatch& m = *it;
    std::string ttype = m[1].str();
    std::vector<int> dims = ParseDimensionInts(m[2].str());
    if (dims.empty()) continue;
    Detection det = DetectDimensions(config_, old_batch_size_, dims);
    auto new_dims =
        CalculateNewDimensions(config_, old_batch_size_, new_batch_size_, dims,
                               det);
    if (!new_dims.has_value()) continue;
    reps.push_back({static_cast<size_t>(m.position(0)),
                    static_cast<size_t>(m.length(0)),
                    absl::StrCat(ttype, "[", absl::StrJoin(*new_dims, ","),
                                   "]")});
  }
  std::sort(reps.begin(), reps.end(),
            [](const Rep& a, const Rep& b) { return a.pos > b.pos; });
  std::string out = line;
  for (const Rep& r : reps) {
    out.replace(r.pos, r.len, r.text);
  }
  return out;
}

absl::StatusOr<std::string> BatchSizeModifier::Modify() {
  if (strict_mode_ && !undetected_msgs_.empty()) {
    std::string msg = "Strict mode: undetected operations:\n";
    for (size_t i = 0; i < std::min(size_t{10}, undetected_msgs_.size());
         ++i) {
      absl::StrAppend(&msg, "  ", undetected_msgs_[i], "\n");
    }
    return absl::InvalidArgumentError(msg);
  }

  std::vector<std::string> modified_lines;
  modified_lines.reserve(lines_.size());

  bool in_metadata = false;
  for (size_t i = 0; i < lines_.size(); ++i) {
    int line_num = static_cast<int>(i + 1);
    const std::string& line = lines_[i];
    std::string stripped = std::string(absl::StripAsciiWhitespace(line));
    if (IsMetadataHeaderLine(line)) {
      in_metadata = true;
      modified_lines.push_back(line);
      continue;
    }
    if (in_metadata) {
      if (stripped.empty()) {
        in_metadata = false;
        modified_lines.push_back(line);
        continue;
      }
      if (!absl::StartsWith(stripped, "%") &&
          !absl::StartsWith(stripped, "HloModule")) {
        modified_lines.push_back(line);
        continue;
      }
      in_metadata = false;
    }

    std::string out = ReplaceConstantsAndSlices(line);
    out = ReplaceAllTensorBracketDims(out);

    if (comm_result_replacements_.contains(line_num)) {
      const auto& rep = comm_result_replacements_.at(line_num);
      out = ReplaceTensorDimsInLine(out, rep.tensor_type, rep.old_dims,
                                    rep.new_dims);
    }
    if (backward_line_map_.contains(line_num)) {
      const BackwardInstInfo& inst = backward_line_map_.at(line_num);
      Detection det =
          DetectDimensions(config_, old_batch_size_, inst.dimensions);
      auto new_dims =
          CalculateNewDimensions(config_, old_batch_size_, new_batch_size_,
                                 inst.dimensions, det);
      if (new_dims.has_value()) {
        out = ReplaceTensorDimsInLine(out, inst.tensor_type, inst.dimensions,
                                      *new_dims);
      }
    }

    modified_lines.push_back(out);
  }

  if (enable_reshape_fix_) {
    modified_lines = FixReshapeOperations(modified_lines);
  }

  return absl::StrJoin(modified_lines, "\n");
}

std::vector<std::string> BatchSizeModifier::FixReshapeOperations(
    const std::vector<std::string>& lines) {
  absl::flat_hash_map<std::string, std::vector<int>> var_to_dims;
  static const std::regex def_with_type(
      R"(%([\w.]+)\s*=\s*[^=]*?([a-zA-Z0-9]+)\[([^\]]+)\])");

  auto refresh_var_map = [&](const std::vector<std::string>& ls) {
    var_to_dims.clear();
    for (const std::string& ln : ls) {
      std::smatch m;
      if (std::regex_search(ln, m, def_with_type)) {
        std::vector<int> dims = ParseDimensionInts(m[3].str());
        if (!dims.empty()) {
          var_to_dims[m[1].str()] = dims;
        }
      }
    }
  };

  std::vector<std::string> result = lines;
  refresh_var_map(result);

  for (size_t line_idx = 0; line_idx < result.size(); ++line_idx) {
    std::string line = result[line_idx];
    if (line.find("reshape") == std::string::npos) continue;
    static const std::regex reshape_op(R"(reshape\(%([\w.]+)\))");
    std::smatch rm;
    if (!std::regex_search(line, rm, reshape_op)) continue;
    std::string op_var = rm[1].str();
    if (!var_to_dims.contains(op_var)) continue;

    std::vector<int> operand_dims = var_to_dims[op_var];
    int64_t op_prod = Product(operand_dims);

    static const std::regex first_tensor(R"(([a-zA-Z0-9]+)\[([^\]]+)\])");
    std::smatch tm;
    if (!std::regex_search(line, tm, first_tensor)) continue;
    std::string rtype = tm[1].str();
    std::vector<int> result_dims = ParseDimensionInts(tm[2].str());
    if (result_dims.empty()) continue;

    int64_t res_prod = Product(result_dims);
    if (res_prod == op_prod) continue;

    if (result_dims.size() == 1) {
      std::regex r(absl::StrCat("(", rtype, R"(\[)(\d+)(\]))"));
      result[line_idx] =
          std::regex_replace(line, r, absl::StrCat("$1", op_prod, "$3"));
      refresh_var_map(result);
      continue;
    }

    if (result_dims.size() == 3) {
      int effective_seq = config_.seq_len;
      if (config_.sp > 1) effective_seq = config_.seq_len / config_.sp;
      int middle = result_dims[1];
      int last = result_dims[2];
      int first = result_dims[0];
      bool seq_match =
          (last == config_.seq_len) ||
          (config_.sp > 1 && last == effective_seq);
      std::vector<int> new_dims = result_dims;
      if (seq_match) {
        if (std::abs(first - old_batch_size_) <= 1 ||
            first == old_batch_size_) {
          new_dims[0] = new_batch_size_;
        } else if (operand_dims.size() >= 2 &&
                   operand_dims[1] == last &&
                   operand_dims[0] > 0) {
          int inferred = operand_dims[0] / old_batch_size_;
          if (inferred == middle || std::abs(inferred - middle) <= 1) {
            new_dims[0] = new_batch_size_;
          } else {
            double scale = static_cast<double>(op_prod) / res_prod;
            new_dims[0] = static_cast<int>(std::llround(first * scale));
            int64_t cur = Product(new_dims);
            if (cur != op_prod && middle > 0 && last > 0) {
              new_dims[0] = static_cast<int>(op_prod / (middle * last));
            }
          }
        } else {
          double scale = static_cast<double>(op_prod) / res_prod;
          new_dims[0] = static_cast<int>(std::llround(first * scale));
          int64_t cur = Product(new_dims);
          if (cur != op_prod && middle > 0 && last > 0) {
            new_dims[0] = static_cast<int>(op_prod / (middle * last));
          }
        }
      } else {
        double scale = static_cast<double>(op_prod) / res_prod;
        new_dims[0] = static_cast<int>(std::llround(first * scale));
        int64_t cur = Product(new_dims);
        if (cur != op_prod && new_dims[0] > 0) {
          int64_t rest = cur / new_dims[0];
          if (rest > 0) new_dims[0] = static_cast<int>(op_prod / rest);
        }
      }
      result[line_idx] = ReplaceTensorDimsInLine(line, rtype, result_dims,
                                                 new_dims);
      std::smatch vm;
      if (std::regex_search(result[line_idx], vm,
                            std::regex(R"(%([\w.]+)\s*=)"))) {
        var_to_dims[vm[1].str()] = new_dims;
      }
      continue;
    }

    if (result_dims.size() > 1) {
      double scale = static_cast<double>(op_prod) / res_prod;
      std::vector<int> new_dims = result_dims;
      int max_dim = *std::max_element(result_dims.begin(), result_dims.end());
      for (size_t j = 0; j < new_dims.size(); ++j) {
        if (j == 0 || new_dims[j] == max_dim) {
          new_dims[j] = static_cast<int>(std::llround(new_dims[j] * scale));
        }
      }
      int64_t cur = Product(new_dims);
      if (cur != op_prod && new_dims[0] > 0) {
        int64_t rest = cur / new_dims[0];
        if (rest > 0) new_dims[0] = static_cast<int>(op_prod / rest);
      }
      result[line_idx] = ReplaceTensorDimsInLine(line, rtype, result_dims,
                                                 new_dims);
      refresh_var_map(result);
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
