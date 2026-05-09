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

#include <stdint.h>

/* Same as analytical_latency_calculator_run but with pipeline-parallelism
   modeling for the Calcium (q250) architecture only. The HLO module is
   interpreted as one pipeline stage; the calculator multiplies by the stage
   count and accounts for inter-stage activation handoff cost.

   PIPELINE PARAMETERS
   -------------------
   num_pipeline_stages: number of pipeline stages. Use 1 (or 0, treated as 1)
       to disable PP modeling; outputs are then byte-identical to
       analytical_latency_calculator_run().
   pipeline_activation_bytes: byte size of the activation handed off between
       consecutive stages. 0 = auto-infer from the HLO entry-computation
       parameter shapes. For training (forward + backward), pass
       2 * forward_size explicitly.
   pipeline_microbatches: number of microbatches per pipeline step. 0 means
       only per-microbatch steady-state metrics are reported (T_total_us is
       left at 0 in pipeline_stats.csv).
   pipeline_comm_overlap_factor: inter-stage compute/handoff overlap factor
       in [0.0, 1.0]. 0.0 = no overlap (conservative; T_step backs up to the
       full slowest handoff). 0.5 = typical 1F1B-style overlap with the next
       stage's compute. 1.0 = handoff fully hidden in cadence. Out-of-range
       values are clamped. Only the steady-state cadence (T_step) benefits
       from this; T_first / bubble / per-microbatch traversal cost use the
       raw handoff because microbatch 0 cannot hide its own handoffs.
   hierarchical_allreduce_enabled: tri-state selector for the Step 9
       Calcium hierarchical AllReduce cost model. Calcium-only; ignored for
       every other arch.
         -1 = leave at built-in OFF default (byte-stable; flat worst-pair AR)
          0 = explicit OFF (flat worst-pair AR)
          1 = ON  (hierarchical {Pod, Card, Server} or {Pod, Rack}
              decomposition - see cebuq/HIERARCHICAL_ALLREDUCE_PLAN.md)
       There is intentionally no .config file key for this flag - this
       parameter (or the matching --hierarchical-allreduce-enabled CLI
       flag) is the only way to turn the model on. This forces the AR
       cost-model selection to be an explicit per-call decision.

   The handoff cost is computed PER BOUNDARY: for each k in [0, S-2] the
   calculator walks the Calcium fabric path between rank-0 of stage k and
   rank-0 of stage k+1, takes the bottleneck EffectiveBandwidth (sharing=1),
   adds 1-us-per-hop latency, and aggregates max/sum into pipeline_stats.csv
   columns t_handoff_max_us / t_handoff_sum_us / t_handoff_us (avg).

   ARCH RESTRICTION
   ----------------
   Pipeline parallelism modeling is currently only supported for q250/calcium.
   Passing num_pipeline_stages > 1 with any other architecture (tpu, b200,
   b300, r200, r576, rcpx) returns non-zero with an explicit error message in
   error_buffer. Other architectures assume PP=1: pass num_pipeline_stages=1
   (or use the legacy analytical_latency_calculator_run()) for those.

   ABI STABILITY
   -------------
   The signature gained pipeline_comm_overlap_factor (Step 8b) and
   hierarchical_allreduce_enabled (Step 9); pre-existing callers of
   _with_pipeline need to recompile. The legacy
   analytical_latency_calculator_run() symbol above is unchanged and
   forwards here with pipeline_comm_overlap_factor=0.0 and
   hierarchical_allreduce_enabled=-1 (leave the AR cost model at its OFF
   default), so callers that only use _run continue to work unmodified
   AND keep the byte-stable flat AR cost behavior.

   When num_pipeline_stages <= 1, this function delegates to the same
   underlying core logic as analytical_latency_calculator_run() - the legacy
   CSV outputs are byte-identical. When num_pipeline_stages > 1, an additional
   pipeline_stats.csv is written under output_dir for each Calcium arch
   processed. */
int analytical_latency_calculator_run_with_pipeline(
    const char* hlo_module_file,
    const char* hardware_architectures,
    const char* output_dir,
    const char* gpu_model_data_root,
    const char* mesh_shape,
    double overlap_factor,
    int fix_ragged_dot_flops,
    int dump_modified_module,
    double scale_memory_bandwidth,
    int num_pipeline_stages,
    int64_t pipeline_activation_bytes,
    int pipeline_microbatches,
    double pipeline_comm_overlap_factor,
    int hierarchical_allreduce_enabled,
    char* error_buffer,
    size_t error_buffer_size);

#ifdef __cplusplus
}
#endif

#endif  /* XLA_SERVICE_GPU_MODEL_ANALYTICAL_LATENCY_CALCULATOR_C_H_ */
