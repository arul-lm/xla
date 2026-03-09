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

#ifndef XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_C_H_
#define XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_C_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C API for use from Rust (or other FFI). Load the shared library and call
   analytical_latency_calculator_run with the parameters below.
   All string parameters must be UTF-8 and null-terminated unless noted. */

/* Runs the analytical latency calculation: loads the HLO module, runs
   cost/communication analysis for each hardware architecture, and writes
   CSV outputs under output_dir (absolute path, or relative to current directory).

   All string parameters are required and must be non-NULL and non-empty
   unless noted.

   hlo_module_file: path to the module file. Format is inferred from the file
   extension. Use .hlo or .txt for HLO text; .mlir is also treated as HLO text
   (the file content should be a HLO variant, not raw MLIR). For actual MLIR
   use .mhlo or .stablehlo. Examples: .../fp4_.../deepseek_r1.mlir,
   .../fp8_.../deepseek_r1.mlir (both HLO content with .mlir extension).
   hardware_architectures: comma-separated list (e.g. "h100_pcie,b200l200").
   output_dir: directory for CSV files (absolute or relative to current directory).
   gpu_model_data_root: root for model data (specs *.txtpb and cluster *.config).
   mesh_shape: exactly 3 positive integers, comma-separated (e.g. "4,4,4").
   overlap_factor: 0.0 to 1.0 (compute-communication overlap).
   fix_ragged_dot_flops: 0 = false, non-zero = true.
   dump_modified_module: 0 = false, non-zero = true.
   scale_memory_bandwidth: scale factor for device memory bandwidth (default 1.0
   = no change). Use e.g. 10.0 to simulate 10x bandwidth; if total time drops
   a lot, the workload is memory-bound. Must be > 0. See "How scale_memory_bandwidth
   is applied" below.

   error_buffer: optional buffer for error message on failure; may be NULL.
   error_buffer_size: size of error_buffer; ignored if error_buffer is NULL.
   If provided, the message is null-terminated and truncated to fit.

   Returns: 0 on success. Non-zero on error; then error_buffer (if non-NULL
   and size > 0) holds the error message.

   How scale_memory_bandwidth is applied:
   Before building the device description for each hardware architecture, the
   loader reads memory_bandwidth from the spec (GpuDeviceInfoProto). If
   scale_memory_bandwidth is set and not 1.0, it replaces memory_bandwidth with
   (memory_bandwidth * scale_memory_bandwidth). That scaled value is then used
   for all read/write time estimates (ReadTimeWithDRAMHeuristic, WriteTime) in
   the performance model, so higher scale = faster memory = lower latency when
   the workload is memory-bound. */
int analytical_latency_calculator_run(
    const char* hlo_module_file,
    const char* hardware_architectures,
    const char* output_dir,
    const char* gpu_model_data_root,
    const char* mesh_shape,
    double overlap_factor,
    int fix_ragged_dot_flops,
    int dump_modified_module,
    double scale_memory_bandwidth,
    char* error_buffer,
    size_t error_buffer_size);

#ifdef __cplusplus
}
#endif

#endif  /* XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_C_H_ */
