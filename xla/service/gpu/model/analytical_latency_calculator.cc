// CLI entry point; core logic lives in analytical_latency_calculator_core.
#include "xla/service/gpu/model/analytical_latency_calculator_core.h"

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "llvm/Support/raw_ostream.h"
#include "tsl/platform/init_main.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/util/command_line_flags.h"

using CliOpts = xla::gpu::AnalyticalLatencyCalculatorOpts;

int main(int argc, char *argv[]) {
  llvm::errs().tie(&llvm::outs());
  CliOpts opts;
  std::string hardware_architectures_str;
  std::string output_path_prefix = ".";
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo-module-file", &opts.hlo_module_file,
                "Filename of HloModule"),
      tsl::Flag("hardware-architectures", &hardware_architectures_str,
                "Comma-separated list of hardware architectures (e.g., "
                "h100_pcie,tpuv5p,tpuv6e)"),
      tsl::Flag("output-dir", &opts.output_dir,
                "Output directory for CSV files (default: stats)"),
      tsl::Flag("output-path-prefix", &output_path_prefix,
                "Base path for output; use /xla when running inside Docker "
                "(default: .)"),
      tsl::Flag("mesh-shape", &opts.mesh_shape,
                "3D mesh shape for TPU communication (e.g., 4,4,4)"),
      tsl::Flag(
          "overlap-factor", &opts.overlap_factor_str,
          "Compute-communication overlap factor (0.0-1.0, default: 0.0)"),
      tsl::Flag("fix-ragged-dot-flops", &opts.fix_ragged_dot_flops,
                "Enable fix for ragged_dot FLOP calculation (default: false)"),
      tsl::Flag("dump-modified-module", &opts.dump_modified_module,
                "Dump modified HLO module to file (default: false)")};
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }

  // Parse hardware architectures from comma-separated string
  if (!hardware_architectures_str.empty()) {
    std::vector<std::string> arch_list =
        absl::StrSplit(hardware_architectures_str, ',');
    for (std::string &arch : arch_list) {
      arch = std::string(absl::StripAsciiWhitespace(arch));
      if (!arch.empty()) {
        opts.hardware_architectures.push_back(arch);
      }
    }
  }

  // Parse overlap_factor from string
  if (!opts.overlap_factor_str.empty()) {
    char *end_ptr;
    opts.overlap_factor =
        std::strtod(opts.overlap_factor_str.c_str(), &end_ptr);
    if (*end_ptr != '\0' || opts.overlap_factor < 0.0 ||
        opts.overlap_factor > 1.0) {
      llvm::outs() << "Error: Overlap factor must be a valid number between "
                      "0.0 and 1.0, got: "
                   << opts.overlap_factor_str << "\n";
      return 1;
    }
  }

  // Parse mesh_shape from comma-separated string
  std::vector<int> mesh_shape;
  if (!opts.mesh_shape.empty()) {
    std::vector<absl::string_view> mesh_parts =
        absl::StrSplit(opts.mesh_shape, ',');
    for (const absl::string_view &part : mesh_parts) {
      absl::string_view trimmed_part = absl::StripAsciiWhitespace(part);
      if (!trimmed_part.empty()) {
        bool is_valid = true;
        for (size_t i = 0; i < trimmed_part.size(); ++i) {
          if (i == 0 && (trimmed_part[i] == '-' || trimmed_part[i] == '+'))
            continue;
          if (!std::isdigit(trimmed_part[i])) {
            is_valid = false;
            break;
          }
        }
        if (!is_valid) {
          llvm::outs() << "Error: Invalid mesh shape value: " << trimmed_part
                       << "\n";
          return 1;
        }
        int value = std::stoi(std::string(trimmed_part));
        if (value <= 0) {
          llvm::outs()
              << "Error: Mesh shape values must be positive integers, got: "
              << value << "\n";
          return 1;
        }
        mesh_shape.push_back(value);
      }
    }
  }

  if (mesh_shape.size() != 3) {
    llvm::outs() << "Error: Mesh shape must have exactly 3 dimensions, got: "
                 << mesh_shape.size() << "\n";
    return 1;
  }

  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);

  absl::Status status = xla::gpu::RunAnalyticalLatencyCalculation(
      opts, mesh_shape, output_path_prefix);
  if (!status.ok()) {
    llvm::outs() << "Error: " << status.message() << "\n";
    llvm::outs().flush();
    return 1;
  }
  return 0;
}
