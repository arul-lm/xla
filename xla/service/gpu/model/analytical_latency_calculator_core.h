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

#ifndef XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_CORE_H_
#define XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_CORE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"

namespace xla {
namespace gpu {

// Options for running the analytical latency calculation. Can be populated
// from CLI flags or from library (FFI) callers. No defaults: all fields must
// be set by the caller.
struct AnalyticalLatencyCalculatorOpts {
  std::string hlo_module_file;
  std::vector<std::string> hardware_architectures;
  std::string output_dir;
  std::string mesh_shape;
  std::string overlap_factor_str;
  double overlap_factor;
  bool fix_ragged_dot_flops;
  bool dump_modified_module;
  /** Root for model data (specs + cluster configs). Required. */
  std::string gpu_model_data_root;
  /**
   * Scale factor for memory bandwidth (default 1.0). Used to verify
   * memory-bound behavior: e.g. 10.0 = 10x bandwidth → if total time
   * drops a lot, the workload is memory-bound.
   */
  double scale_memory_bandwidth = 1.0;

  // Pipeline parallelism (Calcium-only). When num_pipeline_stages <= 1 (the
  // default), all PP code paths are dead and outputs are byte-identical to
  // pre-PP behavior for every architecture.
  int     num_pipeline_stages = 1;          // 1 = no PP modeling
  int64_t pipeline_activation_bytes = 0;    // 0 = auto-infer from HLO
  int     pipeline_microbatches = 0;        // 0 = per-microbatch only
};

// Runs the analytical latency calculation: loads the HLO module, runs
// cost/communication analysis for each hardware architecture, and writes
// CSV outputs (device_stats, comp_stats, comm_stats, overlap_stats,
// instruction_timeline) under output_path_prefix/opts.output_dir.
//
// mesh_shape must have exactly 3 positive integers (e.g. {4, 4, 4}).
// output_path_prefix is the base path for output (e.g. "/xla" in Docker or
// "." for local/FFI). If output_dir is absolute it is used as-is.
//
// Validates opts (e.g. non-empty hlo_module_file, hardware_architectures)
// internally.
absl::Status RunAnalyticalLatencyCalculation(
    const AnalyticalLatencyCalculatorOpts& opts,
    const std::vector<int>& mesh_shape,
    const std::string& output_path_prefix);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_CORE_H_
