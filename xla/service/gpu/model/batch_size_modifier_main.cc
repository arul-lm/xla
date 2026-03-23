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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "tsl/platform/init_main.h"
#include "xla/service/gpu/model/batch_size_modifier_core.h"
#include "xla/tsl/util/command_line_flags.h"

int main(int argc, char* argv[]) {
  std::string input_path;
  std::string output_path;
  std::string config_path;
  std::string mesh_path;
  std::string comm_csv;
  std::string comp_csv;
  int old_batch = 0;
  int new_batch = 0;
  bool no_reshape_fix = false;
  bool strict = false;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input_path, "Input MLIR/HLO-text path (required)"),
      tsl::Flag("output", &output_path, "Output path (required)"),
      tsl::Flag("old-batch-size", &old_batch, "Original batch size (required)"),
      tsl::Flag("new-batch-size", &new_batch, "Target batch size (required)"),
      tsl::Flag("config", &config_path,
                "YAML with modify_batch_size section (required)"),
      tsl::Flag("mesh-inference-path", &mesh_path,
                "Path containing NxMxK for mesh (default: same as --input)"),
      tsl::Flag("comm-stats-csv", &comm_csv,
                "Optional comm_stats.csv (instruction_name, opcode, ...)"),
      tsl::Flag("comp-stats-csv", &comp_csv,
                "Optional comp_stats.csv (instruction_name, opcode, ...)"),
      tsl::Flag("no-reshape-fix", &no_reshape_fix,
                "Disable reshape product consistency pass"),
      tsl::Flag("strict", &strict,
                "Fail if a collective operand/result is not classified"),
  };

  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok || argc != 1) {
    llvm::errs() << usage << "\n";
    return 1;
  }
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (input_path.empty() || output_path.empty() || config_path.empty() ||
      old_batch <= 0 || new_batch <= 0) {
    llvm::errs() << "Error: --input, --output, --config, --old-batch-size, "
                    "--new-batch-size are required with positive batch sizes.\n";
    return 1;
  }

  xla::gpu::BatchSizeModifierOptions opts;
  opts.input_mlir_path = input_path;
  opts.output_mlir_path = output_path;
  opts.old_batch_size = old_batch;
  opts.new_batch_size = new_batch;
  opts.config_yaml_path = config_path;
  if (!mesh_path.empty()) {
    opts.path_for_mesh_inference = mesh_path;
  }
  opts.comm_stats_csv_path = comm_csv;
  opts.comp_stats_csv_path = comp_csv;
  opts.enable_reshape_fix = !no_reshape_fix;
  opts.strict_mode = strict;

  absl::Status st = xla::gpu::RunBatchSizeModification(opts);
  if (!st.ok()) {
    llvm::errs() << absl::StrCat(st.message(), "\n");
    return 1;
  }
  return 0;
}
