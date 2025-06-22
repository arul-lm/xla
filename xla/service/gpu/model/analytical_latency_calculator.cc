#include "gpu_hlo_cost_analysis.h"
#include "iostream"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "llvm/Support/raw_ostream.h"
#include "tsl/platform/init_main.h"
#include "xla/tsl/platform/env.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "absl/strings/match.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/sol_latency_estimator.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"
#include "xla/hlo/ir/hlo_opcode.h"

struct CliOpts {
  std::string hlo_module_file;
};

stream_executor::DeviceDescription rocmAMDMI210DeviceInfo() {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(
      stream_executor::RocmComputeCapability("gfx90a"));
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(64);
  b.set_shared_memory_per_block(64 * 1024);
  b.set_shared_memory_per_block_optin(0);
  b.set_shared_memory_per_core(64 * 1024);
  b.set_threads_per_core_limit(2048);
  b.set_core_count(104);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(2'147'483'647);
  b.set_block_dim_limit_z(2'147'483'647);
  b.set_memory_bandwidth(1'638'400'000'000);
  b.set_l2_cache_size(8 * 1024 * 1024);
  b.set_clock_rate_ghz(1.7);
  b.set_device_memory_size(67'628'957'696);
  b.set_runtime_version(stream_executor::SemanticVersion{6, 0, 0});
  b.set_driver_version(stream_executor::SemanticVersion{6, 0, 0});
  return b;
}

stream_executor::DeviceDescription cudaRTXH100SXMDeviceInfo(
    stream_executor::GpuComputeCapability cc) {
  stream_executor::DeviceDescription b;
  b.set_gpu_compute_capability(cc);
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(32);
  b.set_shared_memory_per_block(48 * 1024);
  b.set_shared_memory_per_block_optin(227 * 1024);
  b.set_shared_memory_per_core(228 * 1024);
  b.set_threads_per_core_limit(2048);
  b.set_core_count(132);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(65535);
  b.set_block_dim_limit_z(65535);
  b.set_memory_bandwidth(3'352'320'000'000);
  b.set_l2_cache_size(50 * 1024 * 1024);
  b.set_clock_rate_ghz(1.98);
  b.set_device_memory_size(84'978'434'048);
  b.set_registers_per_core_limit(65536);
  b.set_registers_per_block_limit(65536);
  b.set_runtime_version(stream_executor::SemanticVersion{12, 4, 0});
  b.set_driver_version(stream_executor::SemanticVersion{12, 4, 0});
  return b;
}

// stream_executor::DeviceDescription cudaRTXA6000DeviceInfo(
//     stream_executor::GpuComputeCapability cc) {
//   stream_executor::DeviceDescription b;
//   b.set_gpu_compute_capability(cc);
//   b.set_threads_per_block_limit(1024);
//   b.set_threads_per_warp(32);
//   b.set_shared_memory_per_block(48 * 1024);
//   b.set_shared_memory_per_block_optin(99 * 1024);
//   b.set_shared_memory_per_core(100 * 1024);
//   b.set_threads_per_core_limit(1536);
//   b.set_core_count(84);
//   b.set_fpus_per_core(128);
//   b.set_block_dim_limit_x(2'147'483'647);
//   b.set_block_dim_limit_y(65535);
//   b.set_block_dim_limit_z(65535);
//   b.set_memory_bandwidth(768'096'000'000);
//   b.set_l2_cache_size(6 * 1024 * 1024);
//   b.set_clock_rate_ghz(1.410);
//   b.set_device_memory_size(51'050'250'240);
//   b.set_registers_per_core_limit(65536);
//   b.set_registers_per_block_limit(65536);
//   b.set_runtime_version(stream_executor::SemanticVersion{12, 4, 0});
//   b.set_driver_version(stream_executor::SemanticVersion{12, 4, 0});
//   return b;
// }

// absl::StatusOr<bool> RunScheduler(
//     xla::HloModule* module, const xla::SchedulerConfig& sched_config,
//     std::unique_ptr<xla::LatencyEstimator> latency_estimator =
//         std::make_unique<xla::ApproximateLatencyEstimator>()) {
//   xla::HloCostAnalysis::ShapeSizeFunction shape_size_bytes =
//       [&shape_size_bytes](const xla::Shape& shape) -> int64_t {
//     int64_t shape_size = 0;
//     if (shape.IsTuple()) {
//       for (auto& sub_shape : shape.tuple_shapes()) {
//         shape_size += shape_size_bytes(sub_shape);
//       }
//       return shape_size;
//     }
//     return xla::ShapeUtil::ByteSizeOfElements(shape);
//   };
//   auto async_tracker = std::make_unique<xla::AsyncTracker>(sched_config);
//   auto scheduler_core = std::make_unique<xla::DefaultSchedulerCore>(
//       shape_size_bytes, async_tracker.get(), latency_estimator.get(),
//       sched_config);
//   TF_ASSIGN_OR_RETURN(
//       bool value, xla::LatencyHidingScheduler(
//                       std::move(latency_estimator), std::move(async_tracker),
//                       std::move(scheduler_core), shape_size_bytes)
//                       .Run(module));

//   return value;
// }

absl::StatusOr<stream_executor::GpuTargetConfigProto> parseDeviceInfo() {
  auto xla_gpu_target_config_filename = "h100_sxm.txtpb";
  std::string gpu_target_config_string;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                           xla_gpu_target_config_filename,
                                           &gpu_target_config_string));
  stream_executor::GpuTargetConfigProto gpu_target_config_proto;

  if (!tsl::protobuf::TextFormat::ParseFromString(gpu_target_config_string,
                                                  &gpu_target_config_proto)) {
    return absl::FailedPreconditionError(
        "Failed to parse GpuTargetConfigProto");
  }

  return gpu_target_config_proto;
}

int main(int argc, char* argv[]) {
  llvm::errs().tie(&llvm::outs());
  CliOpts opts;
  std::vector<tsl::Flag> flag_list = {tsl::Flag(
      "hlo-module-file", &opts.hlo_module_file, "Filename of HloModule")};
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }

  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  CHECK(opts.hlo_module_file.empty() == false)
      << "Path to HLO module file required";

  std::string format = "hlo";
  std::unique_ptr<xla::HloModule> hlo_module =
      *xla::LoadModuleFromFile(opts.hlo_module_file, format, {});
  stream_executor::GpuComputeCapability cc =
      stream_executor::CudaComputeCapability(9, 0);
  // auto gpu_device_info = rocmAMDMI210DeviceInfo();
  auto gpu_device_info = cudaRTXH100SXMDeviceInfo(cc);

  auto const SEPARATOR = ",";
  // auto scheduler_config = xla::SchedulerConfig();
  int64_t pointer_size = 8;

  uint64_t memory_limit = xla::gpu::GetSchedulerMemoryLimit(
      *hlo_module, gpu_device_info, pointer_size);
  std::cout << "Mem validation passed. Memory limit(in GB):"
            << memory_limit / 1e9 << "\n";
  auto collective_overlap_limit = 1;
  xla::SchedulerConfig scheduler_config =
      xla::gpu::MakeGPUSchedulerConfig(memory_limit, collective_overlap_limit);

  auto gle_latency_estimator =
      std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);
  auto ale_latency_estimator =
      std::make_unique<xla::gpu::AnalyticalLatencyEstimator>(
          scheduler_config, std::move(gle_latency_estimator), gpu_device_info,
          xla::HloCostAnalysis::DefaultShapeSize,
          hlo_module->entry_computation());
  auto coll_cost_analysis = std::make_unique<xla::gpu::GpuHloCostAnalysis>(
      xla::gpu::GpuHloCostAnalysis::Options{
          xla::gpu::ShapeSizeBytesFunction(pointer_size),
          /*per_second_rates=*/{},
          /*min_latencies_seconds=*/{},
          /*count_multiple_input_accesses=*/true});

  // auto sol_cost_analysis = std::make_unique<xla::gpu::GpuHloCostAnalysis>(
  //     xla::gpu::GpuHloCostAnalysis::Options{
  //         xla::gpu::ShapeSizeBytesFunction(pointer_size),
  //         /*per_second_rates=*/{},
  //         /*min_latencies_seconds=*/{},
  //         /*count_multiple_input_accesses=*/true});

  // TF_CHECK_OK(
  //     hlo_module->entry_computation()->Accept(coll_cost_analysis.get()));

  auto gpu_latency_estimator =
      std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);
  // TF_CHECK_OK(hlo_module->entry_computation()->Accept(sol_cost_analysis.get()));
  // auto sol_latency_estimator_init = xla::gpu::SolLatencyEstimator::Create(
  //     scheduler_config, std::move(gpu_latency_estimator), gpu_device_info,
  //     xla::gpu::ShapeSizeBytesFunction(pointer_size),
  //     hlo_module->entry_computation(), std::move(sol_cost_analysis));

  // if (sol_latency_estimator_init.ok()) {
    // auto sol_latency_estimator = std::move(*sol_latency_estimator_init);
    auto count = 0;
    auto total_latency = 0.0;
    llvm::outs() << "count" << SEPARATOR << "inst_name" << SEPARATOR
                 << "latency\n";
    auto comp_count = 0;
    for (xla::HloComputation* computation : hlo_module->computations()) {
      if (computation->IsEntryComputation()) {
        for (xla::HloInstruction* instr : computation->instructions()) {
          if (instr->opcode() == xla::HloOpcode::kParameter ||
              instr->opcode() == xla::HloOpcode::kConstant ||
              instr->opcode() == xla::HloOpcode::kTuple ||
              instr->opcode() == xla::HloOpcode::kGetTupleElement ||
              instr->opcode() == xla::HloOpcode::kBitcast) {
            // These instructions always have zero costs.
            continue;
          }
          absl::string_view deduplicated_name =
              instr->metadata().deduplicated_name();
          if (deduplicated_name.empty()) {
            deduplicated_name = instr->name();
          }
          auto opcode = instr->opcode();
          auto cost = 0.0;
          // switch (opcode) {
          //   case xla::HloOpcode::kAllGather:
          //   case xla::HloOpcode::kAllReduce: {
          //     // llvm::outs() << "All-Reduce Inst\n";
          //     // llvm::outs().flush();
          //     auto coll_time = xla::gpu::GpuPerformanceWithCollectiveModel::
          //         ComputeCollectiveTime(*instr, coll_cost_analysis.get(),
          //                               gpu_device_info);
          //     cost = absl::ToDoubleMicroseconds(coll_time);
          //     llvm::outs() << absl::StrFormat("%v", coll_time) << "\n";
          //     llvm::outs() << cost << "\n";
          //     llvm::outs().flush();
          //     break;
          //   }
          //   default: {
          //     auto ale_cost = (ale_latency_estimator)->NodeCost(instr);
          //                       // auto sol_cost = (sol_latency_estimator)->NodeCost(instr);
          //     // llvm::outs() << "ALE|SOL:" << ale_cost << "|" << sol_cost <<
          //     // "\n"; llvm::outs().flush();
          //     // cost = sol_cost;
          //     cost = ale_cost;
          //     break;
          //   }
          // };
          cost = ale_latency_estimator->NodeCost(instr);
          if (cost > 0) {
            llvm::outs() << count << SEPARATOR << deduplicated_name;
            llvm::outs().flush();
            count += 1;
            total_latency += cost;
            llvm::outs() << SEPARATOR << cost << "\n";
          }
        }
      }
      comp_count += 1;
    }
    llvm::outs() << "Total Latency in secs:" << total_latency / 1e6 << "\n";
    llvm::outs() << count << SEPARATOR << comp_count << "Done\n";
    llvm::outs().flush();
  // }
  return 0;
}
