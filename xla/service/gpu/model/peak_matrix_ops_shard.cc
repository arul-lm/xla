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

#include "xla/service/gpu/model/peak_matrix_ops_shard.h"

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "xla/service/gpu/model/peak_matrix_ops_shard_c.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {
namespace gpu {

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

int64_t peak_matrix_ops_per_ns_from_hw_arch(const char* hw_arch_txtpb,
                                           int32_t dtype, char* error_buffer,
                                           size_t error_buffer_size) {
  if (hw_arch_txtpb == nullptr) {
    CopyErrorMessage("hw_arch_txtpb is null", error_buffer, error_buffer_size);
    return -1;
  }
  absl::StatusOr<int64_t> result = CalculatePeakMatrixOpsPerNsFromHwArch(
      absl::string_view(hw_arch_txtpb), static_cast<PrimitiveType>(dtype));
  if (!result.ok()) {
    CopyErrorMessage(std::string(result.status().message()), error_buffer,
                    error_buffer_size);
    return -1;
  }
  return *result;
}

}  // extern "C"

absl::StatusOr<int64_t> CalculatePeakMatrixOpsPerNsFromHwArch(
    absl::string_view hw_arch_txtpb, PrimitiveType dtype) {
  stream_executor::GpuTargetConfigProto proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(std::string(hw_arch_txtpb),
                                                  &proto)) {
    return absl::InvalidArgumentError(
        "Failed to parse hw_arch txtpb as GpuTargetConfigProto");
  }

  const stream_executor::GpuDeviceInfoProto& gpu_device_info_pb =
      proto.gpu_device_info();
  absl::StatusOr<stream_executor::DeviceDescription> device_desc_result =
      stream_executor::DeviceDescription::FromProto(gpu_device_info_pb);

  if (!device_desc_result.ok()) {
    return device_desc_result.status();
  }

  stream_executor::DeviceDescription device_desc =
      std::move(device_desc_result).value();
  if (!proto.device_description_str().empty()) {
    device_desc.set_name(proto.device_description_str());
  }

  auto peak_ops_per_ns = GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(device_desc,
                                                              dtype);
  // Enable sparse boost.
  peak_ops_per_ns = peak_ops_per_ns * 2;
  return peak_ops_per_ns;
}

}  // namespace gpu
}  // namespace xla
