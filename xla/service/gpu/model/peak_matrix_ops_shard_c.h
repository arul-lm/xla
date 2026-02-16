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

#ifndef XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_C_H_
#define XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C API for use from Rust (or other FFI). Load libpeak_matrix_ops_shard.so
   and use symbol "peak_matrix_ops_per_ns_from_hw_arch" with signature below. */

/* Computes peak matrix operations per nanosecond for the given hardware
   architecture (GpuTargetConfigProto in text proto form) and datatype.
   FMA is counted as 2 ops (same as GpuPerformanceModelBase).

   hw_arch_txtpb: null-terminated text proto (e.g. contents of b200.txtpb).
   dtype: XLA PrimitiveType enum value (e.g. 11=F32, 16=BF16, 10=F16).
   error_buffer: optional buffer for error message on failure; may be NULL.
   error_buffer_size: size of error_buffer; ignored if error_buffer is NULL.
                     If provided, the message is null-terminated and truncated
                     to fit.

   Returns: peak matrix ops per nanosecond on success (>= 0).
            -1 on error; then error_buffer (if non-NULL and size > 0) holds
            the error message. */
int64_t peak_matrix_ops_per_ns_from_hw_arch(const char* hw_arch_txtpb,
                                           int32_t dtype,
                                           char* error_buffer,
                                           size_t error_buffer_size);

#ifdef __cplusplus
}
#endif

#endif  /* XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_C_H_ */
