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

#include "xla/service/gpu/model/analytical_latency_calculator_c.h"

#include <cstring>
#include <string>
#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xla/service/gpu/model/analytical_latency_calculator_core.h"

namespace {

void CopyErrorMessage(const std::string& message, char* error_buffer,
                      size_t error_buffer_size) {
  if (error_buffer == nullptr || error_buffer_size == 0) return;
  size_t to_copy = (message.size() < error_buffer_size - 1)
                       ? message.size()
                       : error_buffer_size - 1;
  std::memcpy(error_buffer, message.data(), to_copy);
  error_buffer[to_copy] = '\0';
}

}  // namespace

extern "C" {

int analytical_latency_calculator_run(
    const char* hlo_module_file,
    const char* hardware_architectures,
    const char* output_dir,
    const char* gpu_model_data_root,
    const char* mesh_shape,
    double overlap_factor,
    int fix_ragged_dot_flops,
    int dump_modified_module,
    char* error_buffer,
    size_t error_buffer_size) {
  if (hlo_module_file == nullptr || std::strlen(hlo_module_file) == 0) {
    CopyErrorMessage("hlo_module_file is required", error_buffer,
                     error_buffer_size);
    return 1;
  }
  if (mesh_shape == nullptr || std::strlen(mesh_shape) == 0) {
    CopyErrorMessage("mesh_shape is required (e.g. \"4,4,4\")", error_buffer,
                     error_buffer_size);
    return 1;
  }
  if (output_dir == nullptr || std::strlen(output_dir) == 0) {
    CopyErrorMessage("output_dir is required", error_buffer, error_buffer_size);
    return 1;
  }
  if (gpu_model_data_root == nullptr ||
      std::strlen(gpu_model_data_root) == 0) {
    CopyErrorMessage("gpu_model_data_root is required", error_buffer,
                     error_buffer_size);
    return 1;
  }

  xla::gpu::AnalyticalLatencyCalculatorOpts opts;
  opts.hlo_module_file = hlo_module_file;
  opts.output_dir = output_dir;
  opts.mesh_shape = mesh_shape;
  opts.overlap_factor_str = std::to_string(overlap_factor);
  opts.overlap_factor = overlap_factor;
  opts.gpu_model_data_root = gpu_model_data_root;
  opts.fix_ragged_dot_flops = (fix_ragged_dot_flops != 0);
  opts.dump_modified_module = (dump_modified_module != 0);
  std::string output_prefix_str;  // empty: output_dir is absolute or relative to cwd

  if (hardware_architectures != nullptr &&
      std::strlen(hardware_architectures) > 0) {
    std::vector<std::string> arch_list =
        absl::StrSplit(hardware_architectures, ',');
    for (std::string& arch : arch_list) {
      arch = std::string(absl::StripAsciiWhitespace(arch));
      if (!arch.empty()) {
        opts.hardware_architectures.push_back(arch);
      }
    }
  }
  if (opts.hardware_architectures.empty()) {
    CopyErrorMessage("At least one hardware architecture is required",
                     error_buffer, error_buffer_size);
    return 1;
  }

  std::vector<int> mesh_vec;
  std::vector<absl::string_view> mesh_parts = absl::StrSplit(mesh_shape, ',');
  for (const absl::string_view& part : mesh_parts) {
    absl::string_view trimmed = absl::StripAsciiWhitespace(part);
    if (trimmed.empty()) continue;
    bool valid = true;
    for (size_t i = 0; i < trimmed.size(); ++i) {
      if (i == 0 && (trimmed[i] == '-' || trimmed[i] == '+')) continue;
      if (!std::isdigit(trimmed[i])) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      CopyErrorMessage(
          "mesh_shape must be three positive integers (e.g. \"4,4,4\")",
          error_buffer, error_buffer_size);
      return 1;
    }
    int value = std::stoi(std::string(trimmed));
    if (value <= 0) {
      CopyErrorMessage("mesh_shape values must be positive integers",
                       error_buffer, error_buffer_size);
      return 1;
    }
    mesh_vec.push_back(value);
  }
  if (mesh_vec.size() != 3) {
    CopyErrorMessage("mesh_shape must have exactly 3 dimensions",
                     error_buffer, error_buffer_size);
    return 1;
  }

  absl::Status status = xla::gpu::RunAnalyticalLatencyCalculation(
      opts, mesh_vec, output_prefix_str);
  if (!status.ok()) {
    CopyErrorMessage(std::string(status.message()), error_buffer,
                     error_buffer_size);
    return 1;
  }
  return 0;
}

}  // extern "C"
