/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/model/gpu_performance_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

GpuPerformanceModel::GpuPerformanceModel(
    const se::DeviceDescription& device_info,
    HloFusionAnalysisCache& fusion_analysis_cache,
    GpuPerformanceModelCache& gpu_performance_model_cache,
    mlir::MLIRContext* mlir_context)
    : device_info_(device_info),
      fusion_analysis_cache_(fusion_analysis_cache),
      gpu_performance_model_cache_(gpu_performance_model_cache),
      mlir_context_(mlir_context) {};

// Helper function to convert FPU core calculations to tensor core calculations
// based on primitive type
absl::Duration ConvertFPUToTensorCore(absl::Duration compute_time,
                                      const xla::HloInstruction* instr) {
  auto prim_type = instr->shape().element_type();
  switch (prim_type) {
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::BF16: {
      // F16/BF16 operations can use tensor cores with 16x speedup
      compute_time = compute_time / 16.0;
      break;
    }
    case xla::PrimitiveType::F32: {
      // F32 operations can use tensor cores with 8x speedup
      compute_time = compute_time / 8.0;
      break;
    }
    case xla::PrimitiveType::F64: {
      // F64 operations typically don't use tensor cores, but some GPUs support
      // them with 4x speedup for mixed precision
      compute_time = compute_time / 4.0;
      break;
    }
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::U8: {
      // INT8 operations can use tensor cores with 32x speedup
      compute_time = compute_time / 32.0;
      break;
    }
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16: {
      // INT16 operations can use tensor cores with 16x speedup
      compute_time = compute_time / 16.0;
      break;
    }
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32: {
      // INT32 operations can use tensor cores with 8x speedup
      compute_time = compute_time / 8.0;
      break;
    }
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E5M2FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ: {
      // FP8 operations can use tensor cores with 64x speedup
      compute_time = compute_time / 64.0;
      break;
    }
    default: {
      // For other types (PRED, S64, U64, C64, C128), use default FPU core
      // calculation
      compute_time = compute_time /
                     14.9;  // Compute time is calculated for default fpu cores
      break;
    }
  }
  return compute_time;
}

bool IsTpuDevice(const stream_executor::DeviceDescription& device_info) {
  auto name = device_info.name();
  std::string lower_name = absl::AsciiStrToLower(name);
  return absl::StrContains(lower_name, "tpu");
}

bool IsBlackwell(const stream_executor::DeviceDescription& device_info) {
    auto name = device_info.name();
    std::string lower_name = absl::AsciiStrToLower(name);
    return absl::StrContains(lower_name, "b200") ||
           absl::StrContains(lower_name, "b300");
}

bool IsRubin(const stream_executor::DeviceDescription& device_info) {
  auto name = device_info.name();
  std::string lower_name = absl::AsciiStrToLower(name);
  return absl::StrContains(lower_name, "r200") ||
         absl::StrContains(lower_name, "r576") ||
         absl::StrContains(lower_name, "rcpx");
}

// Calcium (q250) is a non-CUDA accelerator with LPDDR memory (308 GB/s) and
// per-SoC peak ~98 TFLOPS FP8. It uses the same matrix-unit dispatch as
// Blackwell/Rubin (peak FLOPS, sparse boost, saturation curve), but with a
// LPDDR-tuned kKAPPA (see ComputeTimeFromPeakMatrixOps).
bool IsCalcium(const stream_executor::DeviceDescription& device_info) {
  auto name = device_info.name();
  std::string lower_name = absl::AsciiStrToLower(name);
  return absl::StrContains(lower_name, "q250") ||
         absl::StrContains(lower_name, "calcium");
}

// Helper function to get Blackwell tensor core multiplier based on data type
// Returns a multiplier relative to bf16 performance (1.0 = bf16 baseline)
double GetBlackwellDataTypeMultiplier(const xla::HloInstruction* instr) {
  if (!instr) return 1.0;

  auto prim_type = instr->shape().element_type();
  switch (prim_type) {
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::BF16:
      return 1.0;  // Base case (bf16)
    case xla::PrimitiveType::F32:
      return 0.5;  // F32 is 2x slower than bf16 on tensor cores
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::U8:
      return 2.0;  // INT8 is 2x faster than bf16 (32x vs 16x speedup from ConvertFPUToTensorCore)
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
      return 1.0;  // INT16 same as bf16
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
      return 0.5;  // INT32 is 2x slower than bf16
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E5M2FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ:
      return 2.0;  // FP8 is 4x faster than bf16 (64x vs 16x speedup from ConvertFPUToTensorCore)
    default:
      return 1.0;  // Default to bf16 performance
  }
}

// Helper function to get Rubin tensor core multiplier based on data type
// Returns a multiplier relative to bf16 performance (1.0 = bf16 baseline)
double GetRubinDataTypeMultiplier(const xla::HloInstruction* instr) {
  if (!instr) return 1.0;

  auto prim_type = instr->shape().element_type();
  switch (prim_type) {
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::BF16:
      return 1.0;  // Base case (bf16)
    case xla::PrimitiveType::F32:
      return 0.5;  // F32 is 2x slower than bf16 on tensor cores
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::U8:
      return 2.0;  // INT8 is 2x faster than bf16
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
      return 1.0;  // INT16 same as bf16
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
      return 0.5;  // INT32 is 2x slower than bf16
    case xla::PrimitiveType::F8E5M2:
    case xla::PrimitiveType::F8E4M3FN:
    case xla::PrimitiveType::F8E4M3B11FNUZ:
    case xla::PrimitiveType::F8E5M2FNUZ:
    case xla::PrimitiveType::F8E4M3FNUZ:
      return 4.0;  // FP8 is 4x faster than bf16 on Rubin
    default:
      return 1.0;  // Default to bf16 performance
  }
}

absl::Duration ComputeBlackwellTime(
    const stream_executor::DeviceDescription& gpu_device_info,
    int64_t flops,
    int64_t num_blocks,
    int64_t num_threads_per_block,
    const xla::HloInstruction* instr) {
    // Uses tensor cores
    // grouped by warp-group MMA instruction that can handle matrices of size 256x256x16
    // Base calculation assumes bf16. Apply multiplicative factor for other types.
    auto max_bf16_ops = 256 * 256 * 16 * gpu_device_info.clock_rate_ghz();
    auto sparse_boost = 2.0f;

    // Get data type multiplier based on instruction's element type
    double data_type_multiplier = GetBlackwellDataTypeMultiplier(instr);
    auto max_ops_with_boost = max_bf16_ops * sparse_boost * data_type_multiplier;
    // Revert to linear: return absl::Nanoseconds(1.0f * flops / max_ops_with_boost);

    // Apply saturation model: super-linear → linear transition
    // B200 saturation model parameters (hardware-tuned)
    constexpr double kU_MAX = 0.80;           // Sustained utilization ceiling
    constexpr double kKAPPA = 0.04;          // Ramp steepness
    constexpr double kF_REF = 1e12;          // 1 TFLOP reference for saturation onset
    constexpr double kU_MIN = 0.1;
    // Calculate utilization based on saturation model
    double utilization = std::max(kU_MIN, kU_MAX * (1.0 - std::exp(-kKAPPA * static_cast<double>(flops) / kF_REF)));
    // double utilization = 1.0;
    // Effective FLOPS per nanosecond = max_ops_with_boost * utilization
    // Note: max_ops_with_boost is in units of ops per nanosecond
    double effective_flops_per_nanosecond = max_ops_with_boost * utilization;
    double compute_time_nanoseconds = static_cast<double>(flops) / effective_flops_per_nanosecond;
    return absl::Nanoseconds(compute_time_nanoseconds);
}

absl::Duration ComputeRubinTime(
    const stream_executor::DeviceDescription& gpu_device_info,
    int64_t flops,
    int64_t num_blocks,
    int64_t num_threads_per_block,
    const xla::HloInstruction* instr) {
  // Same WGMMA model as Blackwell but 2x base throughput and Rubin-specific
  // data type multipliers.
  double rubin_bf16_ops_per_ns =
      2.0 * 256 * 256 * 16 * gpu_device_info.clock_rate_ghz();
  constexpr double kSparseBoost = 2.0;
  double data_type_multiplier = GetRubinDataTypeMultiplier(instr);
  double max_ops_with_boost =
      rubin_bf16_ops_per_ns * kSparseBoost * data_type_multiplier;

  constexpr double kU_MAX = 0.80;
  constexpr double kKAPPA = 0.04;
  constexpr double kF_REF = 1e12;
  constexpr double kU_MIN = 0.1;
  double utilization = std::max(
      kU_MIN,
      kU_MAX * (1.0 - std::exp(-kKAPPA * static_cast<double>(flops) / kF_REF)));
  double effective_flops_per_nanosecond = max_ops_with_boost * utilization;
  double compute_time_nanoseconds =
      static_cast<double>(flops) / effective_flops_per_nanosecond;
  return absl::Nanoseconds(compute_time_nanoseconds);
}

absl::Duration ComputeTimeFromPeakMatrixOps(
    const stream_executor::DeviceDescription& gpu_device_info, int64_t flops,
    const xla::HloInstruction* instr,
    double* out_effective_ops_per_ns = nullptr) {
  // Use compute precision (first operand) for dots so FP4 vs FP8 get different
  // peak matrix ops; result type is often F32/BF16 accumulation and would
  // otherwise make FP4 and FP8 variants look the same.
  xla::PrimitiveType dtype = instr->shape().element_type();
  if ((instr->opcode() == xla::HloOpcode::kDot ||
       instr->opcode() == xla::HloOpcode::kRaggedDot) &&
      instr->operand_count() >= 1) {
    dtype = instr->operand(0)->shape().element_type();
  }

  int64_t peak_ops_per_ns =
      GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(gpu_device_info, dtype);

  // Enable sparse boost.
  peak_ops_per_ns = peak_ops_per_ns * 2;

  // Apply same saturation model as WGMMA path for consistency.
  constexpr double kU_MAX = 0.80;
  constexpr double kF_REF = 1e12;

  // kKAPPA is per-arch. HBM-backed devices (Blackwell, Rubin) use 0.04
  // because their high memory bandwidth (~8 TB/s) lets compute saturate
  // quickly with moderate FLOP counts.
  //
  // Calcium runs LPDDR (308 GB/s, ~26x lower) and would in principle want a
  // smaller kKAPPA: a strict bandwidth-proportional scaling
  // (0.04 * 308/8000) suggests ~0.0015. We do NOT yet have measured Calcium
  // matmul timings to fit a value, so kKAPPA_CALCIUM is held at the HBM
  // default for now -- this aligns small-kernel saturation behavior with
  // every other arch the model covers, at the known cost of over-estimating
  // effective FLOPS for sub-saturation Calcium kernels. The named constant
  // and IsCalcium gate are preserved as the calibration hook for Step 5b
  // (cebuq/INTEGRATION_GUIDE.md): once silicon timings arrive, only this
  // line changes.
  constexpr double kKAPPA_DEFAULT = 0.04;     // HBM-tuned (Blackwell, Rubin)
  constexpr double kKAPPA_CALCIUM = 0.04;     // pending Calcium calibration;
                                              // see comment above
  const double kappa =
      IsCalcium(gpu_device_info) ? kKAPPA_CALCIUM : kKAPPA_DEFAULT;

  // kU_MIN is the small-FLOP utilization floor. Heavily-sharded Calcium
  // workloads (e.g. TP across a full rack) push per-SoC FLOPs well below
  // F_REF and would otherwise pin every small op to the HBM floor of 0.10.
  // Calcium uses a higher floor (0.35) to reflect that its matrix unit
  // retains a larger fraction of peak on tiny per-SoC tiles than HBM GPUs do.
  // This is the kU_MIN counterpart to kKAPPA_CALCIUM and is a calibration
  // hook for Step 5b -- once silicon timings arrive, only this line changes.
  constexpr double kU_MIN_DEFAULT = 0.10;
  constexpr double kU_MIN_CALCIUM = 0.35;
  const double u_min =
      IsCalcium(gpu_device_info) ? kU_MIN_CALCIUM : kU_MIN_DEFAULT;

  double utilization = std::max(
      u_min,
      kU_MAX * (1.0 - std::exp(-kappa * static_cast<double>(flops) / kF_REF)));
  double effective_ops_per_ns = static_cast<double>(peak_ops_per_ns) * utilization;

  // LOG(INFO) << "PeakMatrixOpsPerNs: " << peak_ops_per_ns
  //           << " (device=" << gpu_device_info.name()
  //           << " dtype=" << xla::PrimitiveType_Name(dtype)
  //           << " peak_TFLOPS=" << (peak_ops_per_ns / 1000.0) << ")"
  //           << "EffectiveOpsPerNs: " << effective_ops_per_ns;
  double compute_time_nanoseconds =
      (effective_ops_per_ns > 0) ? static_cast<double>(flops) / effective_ops_per_ns
                                : 0.0;
  if (out_effective_ops_per_ns != nullptr) {
    *out_effective_ops_per_ns = effective_ops_per_ns;
  }
  return absl::Nanoseconds(compute_time_nanoseconds);
}

absl::Duration ComputeTpuTime(
    const stream_executor::DeviceDescription& tpu_device_info, int64_t flops,
    int64_t num_blocks, int64_t num_threads_per_block) {
  // For TPU: num_threads_per_block = MXU size (128x128 = 16384)
  // num_blocks = number of MXUs
  //   int64_t n_active_mxus = num_blocks;
  //   int64_t n_active_elements_per_mxu = num_threads_per_block;

  // int64_t total_active_elements = n_active_mxus * n_active_elements_per_mxu;

  // // Each element performs 2 FLOPS (multiply + add)
  // int64_t flops_per_ns_per_element = tpu_device_info.clock_rate_ghz() * 2;
  // int64_t effective_flops_per_ns =
  //     flops_per_ns_per_element * total_active_elements;
  // std::cout << "FLOPS:" << flops << "\n";
  // std::cout << "FPNE:" << effective_flops_per_ns << "\n";
    int64_t n_active_fpus_per_core = tpu_device_info.fpus_per_core();
    int64_t n_active_core = tpu_device_info.core_count();
    int64_t fpu_count = n_active_core * n_active_fpus_per_core;
    double flop_per_ns_per_fpu = tpu_device_info.clock_rate_ghz() * /*fma:*/ 2;
    auto flop_per_ns = flop_per_ns_per_fpu * fpu_count;
    // std::cout << "active_fpus_per_core:" << n_active_fpus_per_core << "\n";
    // std::cout << "active_core:" << n_active_core << "\n";
    // std::cout << "fpu_count:" << fpu_count << "\n";
    // std::cout << "flop_per_ns" << flop_per_ns << "\n";
  return absl::Nanoseconds(1.0f * flops / flop_per_ns);
}

EstimateRunTimeData GpuPerformanceModel::EstimateRunTimeForInstructionImpl(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  VLOG(8) << "EstimateRunTimeForInstruction: " << instr->name();

  int64_t flops = cost_analysis->flop_count(*instr);
  int64_t bytes_written = cost_analysis->output_bytes_accessed(*instr);

  const auto& fusion_analysis = fusion_analysis_cache_.Get(*instr);
  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis, mlir_context_);
  int64_t num_blocks = launch_dimensions.num_blocks();

  absl::Duration compute_time;
  std::optional<double> effective_matrix_tflops;
  if(IsTpuDevice(device_info_)){
      compute_time = ComputeTpuTime(device_info_, flops, num_blocks, launch_dimensions.num_threads_per_block());
  } else if (IsBlackwell(device_info_)) {
    // Use peak matrix ops from .txtpb (matrix_unit_description) instead of
    // hardcoded WGMMA. ComputeBlackwellTime() kept for reference/fallback.
    double effective_ops_per_ns = 0.0;
    compute_time = ComputeTimeFromPeakMatrixOps(device_info_, flops, instr,
                                                &effective_ops_per_ns);
    if (effective_ops_per_ns > 0) {
      effective_matrix_tflops = effective_ops_per_ns / 1000.0;  // ops/ns -> TFLOPS
    }
  } else if (IsRubin(device_info_)) {
    // Use peak matrix ops from .txtpb (R200/R200L200 have 2x ops_per_clock).
    // ComputeRubinTime() kept for reference/fallback.
    double effective_ops_per_ns = 0.0;
    compute_time = ComputeTimeFromPeakMatrixOps(device_info_, flops, instr,
                                                &effective_ops_per_ns);
    if (effective_ops_per_ns > 0) {
      effective_matrix_tflops = effective_ops_per_ns / 1000.0;  // ops/ns -> TFLOPS
    }
  } else if (IsCalcium(device_info_)) {
    // Calcium (q250): non-CUDA accelerator, peak FLOPS from
    // matrix_unit_description in q250.txtpb. Saturation curve uses
    // LPDDR-tuned kKAPPA inside ComputeTimeFromPeakMatrixOps.
    double effective_ops_per_ns = 0.0;
    compute_time = ComputeTimeFromPeakMatrixOps(device_info_, flops, instr,
                                                &effective_ops_per_ns);
    if (effective_ops_per_ns > 0) {
      effective_matrix_tflops = effective_ops_per_ns / 1000.0;  // ops/ns -> TFLOPS
    }
  } else {
      compute_time =
          ComputeTime(device_info_, flops, num_blocks,
                      launch_dimensions.num_threads_per_block());
      compute_time = ConvertFPUToTensorCore(compute_time, instr);
  }

  CoalescingAnalysis coalescing_analysis =
      CoalescingAnalysis::Create(instr, instr->operands(), fusion_analysis);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const auto [operand_id, operand] : llvm::enumerate(instr->operands())) {
    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());
    int64_t n_bytes_total =
        GetOperandBytesAccessed(cost_analysis, instr, operand);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    bytes_read += n_bytes_total;

    bool coalesced = coalescing_analysis.IsReadCoalesced(operand);
    PrimitiveType element_type = operand->shape().element_type();

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        device_info_, num_blocks, n_bytes_net, n_bytes_total,
        operand->shape().element_type(),
        GetCoalescingUtilizationRate(element_type, device_info_, coalesced));
  }

  absl::Duration write_time = WriteTime(device_info_, bytes_written);
  absl::Duration exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  EstimateRunTimeData runtime_data = {
      flops,     bytes_read, bytes_written, read_time, write_time,
      compute_time, exec_time, effective_matrix_tflops};
  VLOG(3) << "Runtime data for HLO: " << instr->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();
  return runtime_data;
}

EstimateRunTimeData GpuPerformanceModel::EstimateRunTimeForInstruction(
    const HloInstruction* instr, const GpuHloCostAnalysis* cost_analysis) {
  if (auto cached_result_opt = gpu_performance_model_cache_.Get(*instr)) {
    return *cached_result_opt;
  }

  auto runtime_data = EstimateRunTimeForInstructionImpl(instr, cost_analysis);

  gpu_performance_model_cache_.Set(*instr, runtime_data);

  return runtime_data;
}

absl::Duration GpuPerformanceModel::EstimateRunTimeForFusionImpl(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  VLOG(8) << "EstimateRunTimeForFusion, producer: " << producer->name()
          << " consumer: " << consumer->name();

  if (producer_runtime.IsInfinite() || consumer_runtime.IsInfinite()) {
    return absl::InfiniteDuration();
  }

  float utilization_by_this_consumer = 0;
  for (int64_t i = 0; i < consumer->operand_count(); ++i) {
    if (consumer->operand(i) == producer ||
        (consumer->operand(i)->opcode() == HloOpcode::kGetTupleElement &&
         consumer->operand(i)->operand(0) == producer)) {
      utilization_by_this_consumer +=
          cost_analysis->operand_utilization(*consumer, i);
    }
  }

  const auto& fusion_analysis =
      fusion_analysis_cache_.Get(*producer, *consumer);

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis, mlir_context_);

  int64_t flops = producer_runtime.flops * utilization_by_this_consumer +
                  consumer_runtime.flops;

  absl::Duration compute_time =
      ComputeTime(device_info_, flops, launch_dimensions.num_blocks(),
                  launch_dimensions.num_threads_per_block());

  auto fusion_operands = fusion_analysis.fusion().GetParameters();
  CoalescingAnalysis coalescing_analysis = CoalescingAnalysis::Create(
      producer, consumer, fusion_operands, fusion_analysis);

  absl::Duration read_time;
  int64_t bytes_read = 0;
  for (const auto* operand : fusion_operands) {
    int64_t operand_size = cost_analysis->GetShapeSize(operand->shape());

    int64_t n_bytes_total = GetSharedOperandBytesAccessed(
        cost_analysis, producer, consumer, operand);
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
    bytes_read += n_bytes_total;

    bool coalesced = coalescing_analysis.IsReadCoalesced(operand);
    PrimitiveType element_type = operand->shape().element_type();

    VLogOperandRead(operand, n_bytes_total, n_bytes_net, coalesced);

    read_time += ReadTimeWithDRAMHeuristic(
        device_info_, launch_dimensions.num_blocks(), n_bytes_net,
        n_bytes_total, operand->shape().element_type(),
        GetCoalescingUtilizationRate(element_type, device_info_, coalesced));
  }

  int64_t bytes_written = consumer_runtime.bytes_written;
  absl::Duration write_time = consumer_runtime.write_time;

  // Fusing the producer with the consumer fusion will result in a multi-output
  // fusion that writes output of the producer to the main memory. Add producer
  // output to the total memory write time.
  if (producer_writes_side_output) {
    bytes_written += producer_runtime.bytes_written;
    write_time += producer_runtime.write_time;
  }

  auto exec_time =
      CombineComputeAndMemoryAccessTime(compute_time, read_time + write_time);

  VLOG(3) << "Runtime data for producer-consumer fusion:\n"
          << " producer: " << producer->name() << "\n"
          << " consumer: " << consumer->name() << "\n"
          << launch_dimensions.ToString() << "\n"
          << EstimateRunTimeData{flops,     bytes_read, bytes_written,
                                 read_time, write_time, compute_time,
                                 exec_time, std::nullopt}
                 .ToString();

  return exec_time;
}

absl::Duration GpuPerformanceModel::EstimateRunTimeForFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const EstimateRunTimeData& producer_runtime,
    const EstimateRunTimeData& consumer_runtime,
    const GpuHloCostAnalysis* cost_analysis, bool producer_writes_side_output) {
  if (auto fusion_runtime_opt =
          gpu_performance_model_cache_.Get(*producer, *consumer)) {
    return *fusion_runtime_opt;
  }

  auto fusion_runtime = EstimateRunTimeForFusionImpl(
      producer, consumer, producer_runtime, consumer_runtime, cost_analysis,
      producer_writes_side_output);

  gpu_performance_model_cache_.Set(*producer, *consumer, fusion_runtime);
  return fusion_runtime;
}

GpuPerformanceModel::RunTimes GpuPerformanceModel::EstimateRunTimes(
    const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto cache_result = gpu_performance_model_cache_.Get(*producer);
  CHECK(cache_result.has_value())
      << "Producer `" << producer->name()
      << "` not found in cache. This should never happen! HLO module name: "
      << producer->GetModule()->name()
      << " HLO Instruction: " << producer->ToString();
  EstimateRunTimeData producer_runtime = *cache_result;

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (auto fused_consumer : fused_consumers) {
    VLOG(8) << "Fused consumer: " << fused_consumer->name();

    auto cache_result = gpu_performance_model_cache_.Get(*fused_consumer);
    CHECK(cache_result.has_value())
        << "Consumer `" << fused_consumer->name()
        << "` not found in cache. This should never happen! HLO module name: "
        << fused_consumer->GetModule()->name()
        << " HLO Instruction: " << fused_consumer->ToString();
    EstimateRunTimeData consumer_runtime = *cache_result;

    time_unfused += consumer_runtime.exec_time;

    time_fused +=
        EstimateRunTimeForFusion(producer, fused_consumer, producer_runtime,
                                 consumer_runtime, cost_analysis);
  }

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Consumer count: " << fused_consumers.size();
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

GpuPerformanceModel::RunTimes
GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion(
    const HloInstruction* producer, const HloInstruction* consumer,
    const GpuHloCostAnalysis* cost_analysis) {
  EstimateRunTimeData producer_runtime =
      EstimateRunTimeForInstruction(producer, cost_analysis);
  EstimateRunTimeData consumer_runtime =
      EstimateRunTimeForInstruction(consumer, cost_analysis);

  absl::Duration time_unfused = 2 * kKernelLaunchOverhead +
                                producer_runtime.exec_time +
                                consumer_runtime.exec_time;

  absl::Duration time_fused =
      kKernelLaunchOverhead +
      EstimateRunTimeForFusion(producer, consumer, producer_runtime,
                               consumer_runtime, cost_analysis,
                               /*producer_writes_side_output=*/true);

  if (VLOG_IS_ON(8)) {
    LOG(INFO) << "Unfused time: " << time_unfused;
    LOG(INFO) << "Fused time: " << time_fused;
  }

  return {time_unfused, time_fused};
}

void GpuPerformanceModel::RecordEstimatedRunTime(
    HloInstruction* instruction, const GpuHloCostAnalysis* cost_analysis) {
  DCHECK(Cast<const HloFusionInstruction>(instruction)) << "expected fusion";
  DCHECK(cost_analysis != nullptr) << "expected cost analysis";

  EstimateRunTimeData data =
      EstimateRunTimeForInstruction(instruction, cost_analysis);
  double cycles =
      absl::ToDoubleNanoseconds(data.exec_time) * device_info_.clock_rate_ghz();

  auto gpu_config = instruction->backend_config<GpuBackendConfig>();
  CHECK_OK(gpu_config.status()) << instruction->ToString();
  auto reification_cost = gpu_config->add_reification_cost();
  reification_cost->set_end_to_end_cycles(cycles);
  reification_cost->set_compute_time_us(
      absl::ToDoubleMicroseconds(data.compute_time));
  reification_cost->set_memory_access_time_us(
      absl::ToDoubleMicroseconds(data.read_time + data.write_time));
  reification_cost->set_exec_time_us(
      absl::ToDoubleMicroseconds(data.exec_time));
  CHECK_OK(instruction->set_backend_config(*gpu_config));

  VLOG(8) << "RecordEstimatedRunTime: " << instruction->ToString();
}

}  // namespace gpu
}  // namespace xla
