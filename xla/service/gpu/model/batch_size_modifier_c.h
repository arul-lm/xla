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

#ifndef XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_C_H_
#define XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_C_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rewrites HLO-variant module text (often .mlir on disk) to change batch-related
   tensor dimensions. Mirrors the TokenSim modify_batch_size_v4.py behavior.

   All path strings are UTF-8, null-terminated. Optional paths may be NULL or
   empty (comm_stats_csv_path, comp_stats_csv_path, path_for_mesh_inference).

   path_for_mesh_inference: if NULL or empty, input_mlir_path is used to find
   an embedded mesh shape like 1x72x1 in the path (SPxEPxTP).

   enable_reshape_fix: non-zero to run reshape product consistency fix.
   strict_mode: non-zero to fail if any comm operand/result is not classified.

   Returns 0 on success. On failure, returns non-zero and writes a message into
   error_buffer when error_buffer is non-NULL and error_buffer_size > 0. */
int batch_size_modifier_run(
    const char* input_mlir_path, const char* output_mlir_path,
    int old_batch_size, int new_batch_size, const char* config_yaml_path,
    const char* path_for_mesh_inference, const char* comm_stats_csv_path,
    const char* comp_stats_csv_path, int enable_reshape_fix, int strict_mode,
    char* error_buffer, size_t error_buffer_size);

#ifdef __cplusplus
}
#endif

#endif  // XLA_SERVICE_GPU_MODEL_BATCH_SIZE_MODIFIER_C_H_
