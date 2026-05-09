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
  std::string output_path_prefix;
  std::string scale_memory_bandwidth_str = "1.0";
  std::string pipeline_comm_overlap_factor_str = "0.0";
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo-module-file", &opts.hlo_module_file,
                "Path to HLO module file (required)"),
      tsl::Flag("hardware-architectures", &hardware_architectures_str,
                "Comma-separated list of hardware architectures (required)"),
      tsl::Flag("output-dir", &opts.output_dir,
                "Output directory for CSV files (required)"),
      tsl::Flag("output-path-prefix", &output_path_prefix,
                "Base path for output when output-dir is relative (required)"),
      tsl::Flag("gpu-model-data-root", &opts.gpu_model_data_root,
                "Root for model data: specs and cluster configs (required)"),
      tsl::Flag("mesh-shape", &opts.mesh_shape,
                "3D mesh shape, e.g. 4,4,4 (required)"),
      tsl::Flag("overlap-factor", &opts.overlap_factor_str,
                "Compute-communication overlap factor 0.0-1.0 (required)"),
      tsl::Flag("fix-ragged-dot-flops", &opts.fix_ragged_dot_flops,
                "Enable fix for ragged_dot FLOP calculation"),
      tsl::Flag("scale-memory-bandwidth", &scale_memory_bandwidth_str,
                "Scale memory bandwidth by this factor (default 1.0). Use e.g. "
                "10 to verify memory-bound: if total time drops a lot, workload "
                "is memory-bound."),
      tsl::Flag("dump-modified-module", &opts.dump_modified_module,
                "Dump modified HLO module to file"),
      // Pipeline-parallelism flags (Calcium-only). Defaults preserve legacy
      // behavior: num-pipeline-stages=1 disables PP modeling.
      tsl::Flag("num-pipeline-stages", &opts.num_pipeline_stages,
                "Number of pipeline stages (Calcium-only). Default 1 = no "
                "PP modeling. Values > 1 are rejected for non-Calcium archs."),
      tsl::Flag("pipeline-activation-bytes", &opts.pipeline_activation_bytes,
                "Per-stage activation bytes for inter-stage handoff. 0 = "
                "auto-infer from HLO entry-computation parameter shapes."),
      tsl::Flag("pipeline-microbatches", &opts.pipeline_microbatches,
                "Number of microbatches in the pipeline step. 0 reports "
                "per-microbatch metrics only."),
      // Step 8b. 0.0 keeps pre-Step-8b behavior (no overlap modeled).
      tsl::Flag("pipeline-comm-overlap-factor",
                &pipeline_comm_overlap_factor_str,
                "Inter-stage compute/handoff overlap factor in [0.0, 1.0] "
                "(Calcium-only). The visible per-boundary handoff cost is "
                "multiplied by (1 - factor). 0.0 = no overlap (default, "
                "byte-stable). 0.5 = typical 1F1B-style overlap.")};
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }

  // Require all mandatory flags
  if (opts.hlo_module_file.empty()) {
    llvm::outs() << "Error: --hlo-module-file is required\n";
    return 1;
  }
  if (opts.output_dir.empty()) {
    llvm::outs() << "Error: --output-dir is required\n";
    return 1;
  }
  if (output_path_prefix.empty()) {
    llvm::outs() << "Error: --output-path-prefix is required\n";
    return 1;
  }
  if (opts.gpu_model_data_root.empty()) {
    llvm::outs() << "Error: --gpu-model-data-root is required\n";
    return 1;
  }
  if (opts.mesh_shape.empty()) {
    llvm::outs() << "Error: --mesh-shape is required\n";
    return 1;
  }
  if (opts.overlap_factor_str.empty()) {
    llvm::outs() << "Error: --overlap-factor is required\n";
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

  // Parse scale_memory_bandwidth (optional, default 1.0)
  if (!scale_memory_bandwidth_str.empty()) {
    char *end_ptr;
    opts.scale_memory_bandwidth =
        std::strtod(scale_memory_bandwidth_str.c_str(), &end_ptr);
    if (*end_ptr != '\0' || opts.scale_memory_bandwidth <= 0.0) {
      llvm::outs() << "Error: scale-memory-bandwidth must be a positive number, "
                      "got: "
                   << scale_memory_bandwidth_str << "\n";
      return 1;
    }
  }

  // Parse pipeline_comm_overlap_factor (optional, default 0.0). Step 8b.
  if (!pipeline_comm_overlap_factor_str.empty()) {
    char *end_ptr;
    opts.pipeline_comm_overlap_factor =
        std::strtod(pipeline_comm_overlap_factor_str.c_str(), &end_ptr);
    if (*end_ptr != '\0' || opts.pipeline_comm_overlap_factor < 0.0 ||
        opts.pipeline_comm_overlap_factor > 1.0) {
      llvm::outs() << "Error: pipeline-comm-overlap-factor must be a valid "
                      "number between 0.0 and 1.0, got: "
                   << pipeline_comm_overlap_factor_str << "\n";
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

  if (opts.hardware_architectures.empty()) {
    llvm::outs() << "Error: At least one hardware architecture is required "
                    "(--hardware-architectures)\n";
    return 1;
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
