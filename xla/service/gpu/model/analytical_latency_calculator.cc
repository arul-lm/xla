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
#include "xla/service/gpu/backend_configs.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include <fstream>
#include <vector>

struct CliOpts {
  std::string hlo_module_file;
};

void writeCsv(std::ofstream& outputFile,
              std::vector<std::vector<std::string>> data) {
  for (const auto& row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      outputFile << row[i];
      if (i < row.size() - 1) {
        outputFile << ",";
      }
    }
    outputFile << "\n";
  }
}

std::ofstream createCsv(std::string file_name) {
  std::ofstream ofile(file_name, std::ios::out);
  if (!ofile.is_open()) {
    llvm::outs() << "Unable to open device stats file\n";
    llvm::outs().flush();
  }
  return ofile;
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

  std::string prefix = "/xla";
  auto stats_path = tsl::io::JoinPath("/xla", "stats");
  std::ofstream device_stats_csv =
      createCsv(tsl::io::JoinPath(stats_path, "device_stats.csv"));
  auto op_stats_csv = createCsv(tsl::io::JoinPath(stats_path, "op_stats.csv"));
  if (!device_stats_csv.is_open() || !op_stats_csv.is_open()) {
    return -1;
  }

  std::vector<std::vector<std::string>> device_stats_data, op_stats_data;
  std::vector<std::string> device_stats_header = {"device_name", "latency"};
  std::vector<std::string> op_stats_header = {"idx", "inst", "latency(10^-6)",
                                              "device_name"};
  device_stats_data.push_back(device_stats_header);
  op_stats_data.push_back(op_stats_header);
  std::string format = "hlo";
  std::unique_ptr<xla::HloModule> hlo_module =
      *xla::LoadModuleFromFile(opts.hlo_module_file, format, {});

  absl::flat_hash_map<std::string, stream_executor::GpuDeviceInfoProto>
      gpu_specs;
  for (const std::string file_name :
       {"a100_pcie_80", "a100_sxm_40", "a100_sxm_80", "a6000", "h100_pcie",
        "h100_sxm", "p100", "v100", "mi200"}) {
    stream_executor::GpuTargetConfigProto proto;
    std::string spec_string;
    auto path =
        tsl::io::JoinPath("/xla", "xla", "tools", "hlo_opt", "gpu_specs",
                          absl::StrCat(file_name, ".txtpb"));
    absl::Status is_file_read =
        tsl::ReadFileToString(tsl::Env::Default(), path, &spec_string);
    if (is_file_read.ok()) {
      tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto);
    } else {
      llvm::outs() << "Device spec file read failed\n";
      return -1;
    }
    llvm::outs().flush();
    gpu_specs[proto.device_description_str()] = proto.gpu_device_info();
  }

  auto const SEPARATOR = ",";
  int64_t pointer_size = 8;
  for (const auto& pair : gpu_specs) {
    auto device_name = pair.first;
    stream_executor::GpuDeviceInfoProto gpu_device_info_pb = pair.second;
    auto gpu_device_info =
        stream_executor::DeviceDescription(gpu_device_info_pb);
    uint64_t memory_limit = xla::gpu::GetSchedulerMemoryLimit(
        *hlo_module, gpu_device_info, pointer_size);
    std::cout << "Mem validation passed. Memory limit(in GB):"
              << memory_limit / 1e9 << "\n";
    auto collective_overlap_limit = 1;
    xla::SchedulerConfig scheduler_config = xla::gpu::MakeGPUSchedulerConfig(
        memory_limit, collective_overlap_limit);

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
    TF_CHECK_OK(
        hlo_module->entry_computation()->Accept(coll_cost_analysis.get()));

    auto gpu_latency_estimator =
        std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);

    auto count = 0;
    auto total_latency = 0.0;
    llvm::outs() << "count" << SEPARATOR << "inst_name" << SEPARATOR
                 << "latency\n";
    auto comp_count = 0;
    auto comp_cost = 0.0;
    absl::flat_hash_map<const xla::HloComputation*, double> computation_map;
    absl::flat_hash_map<const xla::HloComputation*, int> computation_idx;
    for (xla::HloComputation* computation : hlo_module->computations()) {
      // if (computation->IsEntryComputation()) {
      comp_cost = 0;
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
        switch (opcode) {
          case xla::HloOpcode::kWhile: {
            auto comps = {instr->while_body(), instr->while_condition()};
            auto while_cost = 0.0;
            for (auto comp : comps) {
              auto it = computation_map.find(comp);
              // auto it2 = computation_idx.find(comp);
              // llvm::outs() << "Adding cost of comp:" << it2->second << "\n";
              CHECK(it != computation_map.end());
              while_cost += it->second;
              cost = ale_latency_estimator->NodeCost(instr);
            }
            absl::StatusOr<xla::WhileLoopBackendConfig> while_config =
                instr->backend_config<xla::WhileLoopBackendConfig>();
            if (while_config.ok() && while_config->has_known_trip_count()) {
              auto one_time_cost = ale_latency_estimator->NodeCost(instr);
              cost = while_cost * while_config->known_trip_count().n() +
                     one_time_cost;
            } else {
              CHECK(1 == 0);
            }
            break;
          }
          case xla::HloOpcode::kCollectivePermute:
          case xla::HloOpcode::kAllGather:
          case xla::HloOpcode::kAllReduce: {
            auto coll_time = xla::gpu::GpuPerformanceWithCollectiveModel::
                ComputeCollectiveTime(*instr, coll_cost_analysis.get(),
                                      gpu_device_info);
            cost = absl::ToDoubleMicroseconds(coll_time);
            break;
          }
          default: {
            cost = ale_latency_estimator->NodeCost(instr);
            break;
          }
        };
        comp_cost += cost;
        if (cost > 0) {
          count += 1;
          total_latency += cost;
          op_stats_data.push_back({std::to_string(count), std::string(deduplicated_name),
                                   std::to_string(cost), device_name});
        }
      }
      computation_map.insert({computation, comp_cost});
      computation_idx.insert({computation, comp_count});
      comp_count += 1;
    }
    auto comp_cost_in_secs = comp_cost / 1e6;
    device_stats_data.push_back(
        {device_name, std::to_string(comp_cost_in_secs)});
    llvm::outs().flush();
  }
  writeCsv(device_stats_csv, device_stats_data);
  writeCsv(op_stats_csv, op_stats_data);
  device_stats_csv.close();
  op_stats_csv.close();
  llvm::outs() << "Done\n";
  return 0;
}
