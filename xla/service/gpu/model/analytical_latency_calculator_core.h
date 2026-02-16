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

#include <string>
#include <vector>

#include "absl/status/status.h"

namespace xla {
namespace gpu {

// Options for running the analytical latency calculation. Can be populated
// from CLI flags or from library (FFI) callers.
struct AnalyticalLatencyCalculatorOpts {
  std::string hlo_module_file;
  std::vector<std::string> hardware_architectures;
  std::string output_dir = "stats";
  std::string mesh_shape = "4,4,4";
  std::string overlap_factor_str = "0.0";
  double overlap_factor = 0.0;
  bool fix_ragged_dot_flops = false;
  bool dump_modified_module = false;
  /** Optional root for model data (specs + cluster configs). When set, used for
   * path resolution; empty = use default paths (e.g. /xla in Docker). */
  std::string gpu_model_data_root;
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
