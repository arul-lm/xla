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

#include "xla/service/gpu/model/batch_size_modifier_c.h"

#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "xla/service/gpu/model/batch_size_modifier_core.h"

namespace {

void CopyError(const std::string& message, char* error_buffer,
               size_t error_buffer_size) {
  if (error_buffer == nullptr || error_buffer_size == 0) return;
  size_t n = message.size() < error_buffer_size - 1
                 ? message.size()
                 : error_buffer_size - 1;
  std::memcpy(error_buffer, message.data(), n);
  error_buffer[n] = '\0';
}

}  // namespace

extern "C" {

int batch_size_modifier_run(const char* input_mlir_path,
                            const char* output_mlir_path, int old_batch_size,
                            int new_batch_size, const char* config_yaml_path,
                            const char* path_for_mesh_inference,
                            const char* comm_stats_csv_path,
                            const char* comp_stats_csv_path,
                            int enable_reshape_fix, int strict_mode,
                            char* error_buffer, size_t error_buffer_size) {
  if (input_mlir_path == nullptr || std::strlen(input_mlir_path) == 0) {
    CopyError("input_mlir_path is required", error_buffer, error_buffer_size);
    return 1;
  }
  if (output_mlir_path == nullptr || std::strlen(output_mlir_path) == 0) {
    CopyError("output_mlir_path is required", error_buffer, error_buffer_size);
    return 1;
  }
  if (config_yaml_path == nullptr || std::strlen(config_yaml_path) == 0) {
    CopyError("config_yaml_path is required", error_buffer, error_buffer_size);
    return 1;
  }

  xla::gpu::BatchSizeModifierOptions opts;
  opts.input_mlir_path = input_mlir_path;
  opts.output_mlir_path = output_mlir_path;
  opts.old_batch_size = old_batch_size;
  opts.new_batch_size = new_batch_size;
  opts.config_yaml_path = config_yaml_path;
  if (path_for_mesh_inference != nullptr &&
      std::strlen(path_for_mesh_inference) > 0) {
    opts.path_for_mesh_inference = path_for_mesh_inference;
  }
  if (comm_stats_csv_path != nullptr && std::strlen(comm_stats_csv_path) > 0) {
    opts.comm_stats_csv_path = comm_stats_csv_path;
  }
  if (comp_stats_csv_path != nullptr && std::strlen(comp_stats_csv_path) > 0) {
    opts.comp_stats_csv_path = comp_stats_csv_path;
  }
  opts.enable_reshape_fix = (enable_reshape_fix != 0);
  opts.strict_mode = (strict_mode != 0);

  absl::Status st = xla::gpu::RunBatchSizeModification(opts);
  if (!st.ok()) {
    CopyError(std::string(st.message()), error_buffer, error_buffer_size);
    return 1;
  }
  return 0;
}

}  // extern "C"
