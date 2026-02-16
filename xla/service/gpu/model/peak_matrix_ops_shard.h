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

#ifndef XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_H_
#define XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Computes peak matrix operations per nanosecond for the given hardware
// architecture and datatype.
//
// `hw_arch_txtpb` is the text proto (txtpb) content of a GpuTargetConfigProto
// (e.g. from a .txtpb file such as b200.txtpb or r200.txtpb). It describes
// the GPU device (cores, matrix units, clock rates, etc.).
//
// `dtype` is the XLA primitive type (e.g. F32, BF16, F16) for which to
// compute peak matrix ops. The result uses the same convention as
// GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs (FMA counted as 2 ops).
//
// Returns the peak matrix ops per nanosecond, or an error if the txtpb cannot
// be parsed or converted to a device description.
absl::StatusOr<int64_t> CalculatePeakMatrixOpsPerNsFromHwArch(
    absl::string_view hw_arch_txtpb, PrimitiveType dtype);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_PEAK_MATRIX_OPS_SHARD_H_
