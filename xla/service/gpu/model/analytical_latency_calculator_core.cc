// Core logic shared between CLI and C FFI.
#include "xla/service/gpu/model/analytical_latency_calculator_core.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "cluster_config.h"
#include "gpu_hlo_cost_analysis.h"
#include "gpu_performance_model.h"
#include "gpu_performance_model_base.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"
#include "xla/service/gpu/model/sol_latency_estimator.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/util.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <vector>

// Forward declarations
uint64_t GetNumReplicaGroups(const xla::HloInstruction *instr);
uint64_t GetReplicaGroupSize(const xla::HloInstruction *instr);
std::string GetDataTypeFromInstruction(const xla::HloInstruction *instr);

struct InstructionTimelineEntry {
  const xla::HloInstruction *instruction;
  double start_time_us;
  double end_time_us;
  double duration_us;
  bool is_compute_instruction;
  bool is_comm_instruction;
  std::string instruction_name;
};

struct OverlapStats {
  double original_total_time_us;
  double overlapped_total_time_us;
  double overlap_savings_us;
  double overlap_percentage;
  double total_compute_time_us;
  double total_comm_time_us;
  double overlap_factor;
  std::vector<InstructionTimelineEntry> timeline;
};

// Pipeline-parallelism summary metrics for a single Calcium architecture.
struct PipelineMetrics {
  int    num_pipeline_stages;
  int64_t pipeline_activation_bytes;
  int    pipeline_microbatches;
  int    devices_per_stage;
  double t_stage_us;          // per-stage compute+comm latency (one microbatch)
  // Per-boundary handoff aggregates (Step 8b). t_handoff_us preserves the
  // pre-Step-8b CSV column, set to t_handoff_avg_us so single-stage callers
  // see the average per-boundary cost. T_step uses max; T_first / bubble use
  // sum.
  double t_handoff_us;         // average per-boundary handoff latency
  double t_handoff_max_us;     // slowest boundary; drives T_step
  double t_handoff_sum_us;     // sum over (S - 1) boundaries; drives T_first
  // Inter-stage handoff/compute overlap factor in [0,1] actually applied.
  // Distinct from the within-stage compute/comm `overlap_factor` carried in
  // AnalyticalLatencyCalculatorOpts; they apply at different layers (intra-
  // stage instruction overlap vs inter-stage handoff overlap).
  double pipeline_comm_overlap_factor;
  double t_first_us;           // pipeline fill latency
  double t_step_us;            // steady-state per-microbatch latency
  double t_total_us;           // total runtime for M microbatches (0 if M==0)
  double bubble_time_us;       // pipeline-fill bubble time
  double bubble_fraction;      // bubble_time_us / t_total_us
  std::string bound_by;        // "compute" | "comm" | "balanced"
};

// Sum the byte sizes of all entry-computation parameter shapes. Used as the
// default pipeline activation size when the caller passes 0 in
// AnalyticalLatencyCalculatorOpts::pipeline_activation_bytes.
int64_t InferActivationBytes(const xla::HloModule& hlo_module) {
  int64_t total = 0;
  const xla::HloComputation* entry = hlo_module.entry_computation();
  if (entry == nullptr) return 0;
  for (const xla::HloInstruction* param : entry->parameter_instructions()) {
    total += xla::ShapeUtil::ByteSizeOf(param->shape());
  }
  return total;
}

// Compute pipeline-parallelism metrics from per-microbatch latencies and
// per-boundary handoff aggregates. Forward-only inference schedule with
// physical overlap semantics: pipeline_comm_overlap_factor only hides
// handoff cost in steady-state cadence (T_step), NOT in the path of any
// single microbatch through the pipe (T_first / bubble / TTFT). It is
// distinct from the within-stage compute/comm `overlap_factor` that
// already shaped t_stage_us before this function was called.
//
//   T_first    = num_stages * T_stage + raw_sum     <- mb 0 traversal cost
//   T_step     = max(T_stage, eff_max)              <- with overlap
//   T_total(M) = T_first + (M - 1) * T_step
//   bubble     = (num_stages - 1) * T_stage + raw_sum
//
// where eff_k = raw_k * (1 - pipeline_comm_overlap_factor).
//
// For pipeline_comm_overlap_factor == 0.0, raw == eff and the formulas
// collapse to the legacy T_first = S*T_stage + (S-1)*T_handoff form
// (assuming uniform per-boundary cost). bound_by compares T_stage against
// eff_max because cadence (not traversal) drives "compute vs comm"
// classification.
PipelineMetrics ComputePipelineMetrics(
    int num_pipeline_stages, int64_t pipeline_activation_bytes,
    int pipeline_microbatches, int devices_per_stage,
    double t_stage_us, const PipelineHandoffStats& handoffs) {
  PipelineMetrics m;
  m.num_pipeline_stages = num_pipeline_stages;
  m.pipeline_activation_bytes = pipeline_activation_bytes;
  m.pipeline_microbatches = pipeline_microbatches;
  m.devices_per_stage = devices_per_stage;
  m.t_stage_us = t_stage_us;
  m.t_handoff_us = handoffs.avg_us;
  m.t_handoff_max_us = handoffs.max_us;       // effective max (drives T_step)
  m.t_handoff_sum_us = handoffs.raw_sum_us;   // raw sum (drives T_first)
  m.pipeline_comm_overlap_factor = handoffs.pipeline_comm_overlap_factor;

  const int S = std::max(1, num_pipeline_stages);
  m.t_first_us = static_cast<double>(S) * t_stage_us + handoffs.raw_sum_us;
  m.t_step_us = std::max(t_stage_us, handoffs.max_us);

  if (pipeline_microbatches > 0) {
    m.t_total_us = m.t_first_us +
                   (static_cast<double>(pipeline_microbatches) - 1.0) *
                       m.t_step_us;
  } else {
    m.t_total_us = 0.0;
  }

  m.bubble_time_us =
      static_cast<double>(S - 1) * t_stage_us + handoffs.raw_sum_us;
  m.bubble_fraction = (m.t_total_us > 0.0)
                          ? (m.bubble_time_us / m.t_total_us)
                          : 0.0;

  // bound_by classification: 10% threshold for "balanced". Compare t_stage
  // against the slowest effective boundary (handoff_max), since cadence
  // (T_step) is what defines "compute vs comm" in steady state.
  const double t_handoff_cadence = handoffs.max_us;
  if (t_stage_us <= 0.0 && t_handoff_cadence <= 0.0) {
    m.bound_by = "balanced";
  } else {
    const double max_v = std::max(t_stage_us, t_handoff_cadence);
    const double min_v = std::min(t_stage_us, t_handoff_cadence);
    const double ratio = (max_v > 0.0) ? (min_v / max_v) : 1.0;
    if (ratio >= 0.9) {
      m.bound_by = "balanced";
    } else if (t_stage_us > t_handoff_cadence) {
      m.bound_by = "compute";
    } else {
      m.bound_by = "comm";
    }
  }
  return m;
}

// Helper function to get device IDs from one replica group after symmetry check
std::vector<int64_t>
GetDeviceIdsFromOneReplicaGroup(const xla::HloInstruction *instr) {
  std::vector<int64_t> device_ids;

  // Handle CollectivePermute separately since it doesn't inherit from
  // HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto *collective_permute =
        xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto &pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      device_ids.assign(participants.begin(), participants.end());
    }
    return device_ids;
  }

  // Handle other collective operations that inherit from
  // HloCollectiveInstruction Only try to cast for operations that actually
  // inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto *collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto &replica_groups = collective_inst->replica_groups();

      if (replica_groups.empty()) {
        // If no replica groups, return empty vector
        return device_ids;
      }

      // Check symmetry: all groups should have the same size
      const size_t first_group_size = replica_groups[0].replica_ids_size();
      for (size_t i = 1; i < replica_groups.size(); ++i) {
        CHECK_EQ(replica_groups[i].replica_ids_size(), first_group_size)
            << "Expected symmetric replica groups for instruction: "
            << instr->ToString();
      }

      // Extract device IDs from the first group
      const auto &first_group = replica_groups[0];
      for (size_t i = 0; i < first_group.replica_ids_size(); ++i) {
        device_ids.push_back(first_group.replica_ids(i));
      }
    }
  }

  return device_ids;
}

OverlapStats CalculateInstructionLevelOverlap(
    const std::vector<InstructionTimelineEntry> &timeline,
    double overlap_factor) {
  CHECK_GE(overlap_factor, 0.0)
      << "overlap_factor must be non-negative, got: " << overlap_factor;
  CHECK_LE(overlap_factor, 1.0)
      << "overlap_factor must be <= 1.0, got: " << overlap_factor;

  if (timeline.empty()) {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, overlap_factor, {}};
  }

  double original_total_time_us = 0.0;
  double total_compute_time_us = 0.0;
  double total_comm_time_us = 0.0;

  for (const auto &entry : timeline) {
    original_total_time_us += entry.duration_us;
    if (entry.is_compute_instruction) {
      total_compute_time_us += entry.duration_us;
    }
    if (entry.is_comm_instruction) {
      total_comm_time_us += entry.duration_us;
    }
  }

  if (overlap_factor == 0.0) {
    return {original_total_time_us,
            original_total_time_us,
            0.0,
            0.0,
            total_compute_time_us,
            total_comm_time_us,
            overlap_factor,
            timeline};
  }

  std::vector<InstructionTimelineEntry> overlapped_timeline = timeline;
  double current_time_us = 0.0;
  double last_compute_time_us = 0.0;
  double last_comm_time_us = 0.0;
  for (size_t i = 0; i < overlapped_timeline.size(); ++i) {
    auto &current_instruction = overlapped_timeline[i];

    if (current_instruction.start_time_us == 0.0) {
      // unscheduled instruction
      current_instruction.start_time_us = current_time_us;
      current_instruction.end_time_us =
          current_time_us + current_instruction.duration_us;
    } else {
      // current instruction is already scheduled.
      current_time_us = current_instruction.start_time_us;
    }
    // Try to overlap with next instruction
    if (overlap_factor > 0.0) {
      size_t j = i + 1;
      if (j < overlapped_timeline.size()) {
        auto &next_instruction = overlapped_timeline[j];
        bool can_overlap = (current_instruction.is_compute_instruction &&
                            next_instruction.is_comm_instruction) ||
                           (current_instruction.is_comm_instruction &&
                            next_instruction.is_compute_instruction);
        CHECK_EQ(next_instruction.start_time_us, 0.0)
            << "Next instruction start time is not 0.0";
        if (can_overlap) {
          double overlap_duration =
              current_instruction.duration_us * overlap_factor;
          double next_start_time =
              current_instruction.start_time_us +
              (current_instruction.duration_us - overlap_duration);
          if (next_instruction.is_compute_instruction) {
            next_start_time = std::max(next_start_time, last_compute_time_us);
          } else if (next_instruction.is_comm_instruction) {
            next_start_time = std::max(next_start_time, last_comm_time_us);
          }
          next_instruction.start_time_us = next_start_time;
          next_instruction.end_time_us =
              next_start_time + next_instruction.duration_us;
          current_time_us = next_start_time;
        } else {
          current_time_us = current_instruction.end_time_us;
        }
      } else {
        llvm::outs() << "Nothing to overlap with\n";
        current_time_us = current_instruction.end_time_us;
      }
    }
    if (current_instruction.is_compute_instruction) {
      last_compute_time_us =
          current_instruction.start_time_us + current_instruction.duration_us;
    } else if (current_instruction.is_comm_instruction) {
      last_comm_time_us =
          current_instruction.start_time_us + current_instruction.duration_us;
    }
  }

  double overlapped_total_time_us = 0.0;
  for (const auto &entry : overlapped_timeline) {
    overlapped_total_time_us =
        std::max(overlapped_total_time_us, entry.end_time_us);
  }

  double overlap_savings_us = original_total_time_us - overlapped_total_time_us;
  double overlap_percentage =
      original_total_time_us > 0.0
          ? (overlap_savings_us / original_total_time_us) * 100.0
          : 0.0;

  return {original_total_time_us, overlapped_total_time_us, overlap_savings_us,
          overlap_percentage,     total_compute_time_us,    total_comm_time_us,
          overlap_factor,         overlapped_timeline};
}

// Helper function to convert CommType enum to string
std::string CommTypeToString(CommType comm_type) {
  switch (comm_type) {
  case CommType::ScaleUp:
    return "ScaleUp";
  case CommType::Rail:
    return "Rail";
  case CommType::RailOffset:
    return "RailOffset";
  case CommType::ScaleOut:
    return "ScaleOut";
  default:
    return "Unknown";
  }
}
std::string escapeCsvField(const std::string& field) {
  // Check if field needs quoting (contains comma, newline, or double quote)
  bool needs_quoting = field.find(',') != std::string::npos ||
                       field.find('\n') != std::string::npos ||
                       field.find('"') != std::string::npos;
  
  if (!needs_quoting) {
    return field;
  }
  
  // Escape double quotes by doubling them and wrap in quotes
  std::string escaped;
  escaped.reserve(field.size() + 2);  // Reserve space for quotes
  escaped += '"';
  for (char c : field) {
    if (c == '"') {
      escaped += "\"\"";  // Escape double quote
    } else {
      escaped += c;
    }
  }
  escaped += '"';
  return escaped;
}

// Parse duration string from comp_stats (e.g. "8.64768025ms", "1.265s", "694us") to seconds.
double ParseDurationToSeconds(const std::string &s) {
  if (s.empty()) return 0.0;
  const char *p = s.c_str();
  while (*p == ' ' || *p == '\t') ++p;
  char *end;
  double val = std::strtod(p, &end);
  if (end == p) return 0.0;
  while (*end == ' ' || *end == '\t') ++end;
  if (*end == 'n' && end[1] == 's') return val * 1e-9;
  if (*end == 'u' && end[1] == 's') return val * 1e-6;
  if (*end == 'm' && end[1] == 's') return val * 1e-3;
  if (*end == 's' && (end[1] == '\0' || end[1] == ' ')) return val;
  if (*end == 'm' && end[1] == 'i' && end[2] == 'n') return val * 60.0;
  if (*end == 'h' && (end[1] == '\0' || end[1] == ' ')) return val * 3600.0;
  return val;  // assume seconds if no unit
}

void writeCsv(std::ofstream &outputFile,
              const std::vector<std::vector<std::string>> &data,
              bool include_header = true) {
  size_t start = include_header ? 0 : 1;
  for (size_t r = start; r < data.size(); ++r) {
    const auto &row = data[r];
    for (size_t i = 0; i < row.size(); ++i) {
      outputFile << escapeCsvField(row[i]);
      if (i < row.size() - 1) {
        outputFile << ",";
      }
    }
    outputFile << "\n";
  }
}

struct CsvOut {
  std::ofstream stream;
  bool write_header;
};

CsvOut createCsv(const std::string &file_name, bool append = false) {
  std::ios_base::openmode mode =
      append ? (std::ios::out | std::ios::app) : std::ios::out;
  std::ofstream ofile(file_name, mode);
  bool write_header = true;
  if (ofile.is_open() && append) {
    std::streampos pos = ofile.tellp();
    write_header = (pos == 0);
  }
  if (!ofile.is_open()) {
    llvm::outs() << "Unable to open file: " << file_name << "\n";
    llvm::outs().flush();
  }
  return CsvOut{std::move(ofile), write_header};
}

// Relative paths under gpu_model_data_root.
const char kSpecsRelativePath[] = "xla/backends/gpu/target_config/specs";
const char kConfigsRelativePath[] = "xla/service/gpu/model/configs";

// Resolve the directory containing cluster *.config files.
// Requires opts.gpu_model_data_root to be non-empty.
std::string ResolveConfigsDir(
    const xla::gpu::AnalyticalLatencyCalculatorOpts& opts) {
  return tsl::io::JoinPath(opts.gpu_model_data_root, kConfigsRelativePath);
}

// Resolve the directory containing GPU target config spec files (*.txtpb).
// Requires opts.gpu_model_data_root to be non-empty.
std::string ResolveSpecsDir(
    const xla::gpu::AnalyticalLatencyCalculatorOpts& opts) {
  return tsl::io::JoinPath(opts.gpu_model_data_root, kSpecsRelativePath);
}

// Helper function to validate hardware architectures and check if config files
// exist
absl::Status ValidateHardwareArchitectures(
    const std::vector<std::string> &hardware_architectures,
    const std::string &specs_dir) {
  if (hardware_architectures.empty()) {
    return absl::InvalidArgumentError(
        "No hardware architectures specified. Use --hardware-architectures to "
        "specify at least one architecture.");
  }

  std::vector<std::string> invalid_architectures;

  for (const std::string &arch : hardware_architectures) {
    if (arch.empty()) {
      invalid_architectures.push_back("(empty string)");
      continue;
    }

    std::string config_path =
        tsl::io::JoinPath(specs_dir, absl::StrCat(arch, ".txtpb"));

    absl::Status file_exists_status =
        tsl::Env::Default()->FileExists(config_path);
    if (!file_exists_status.ok()) {
      invalid_architectures.push_back(arch);
    }
  }

  if (!invalid_architectures.empty()) {
    std::string error_msg =
        "Invalid hardware architectures - config files not found:\n";
    for (const std::string &invalid_arch : invalid_architectures) {
      std::string expected_path =
          tsl::io::JoinPath(specs_dir, absl::StrCat(invalid_arch, ".txtpb"));
      error_msg += absl::StrCat("  - ", invalid_arch,
                                " (expected: ", expected_path, ")\n");
    }
    error_msg += "\nSet gpu_model_data_root to the repo root (e.g. /data/home/arul/dev/xla). ";
    error_msg += "Expected specs dir: ";
    error_msg += specs_dir;
    return absl::InvalidArgumentError(error_msg);
  }

  return absl::OkStatus();
}

// Helper function to get number of replica groups from instruction
uint64_t GetNumReplicaGroups(const xla::HloInstruction *instr) {
  // Handle CollectivePermute separately since it doesn't inherit from
  // HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto *collective_permute =
        xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto &pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      // For CollectivePermute, we consider it as one replica group
      return 1;
    }
    return 0;
  }

  // Handle other collective operations that inherit from
  // HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto *collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto &replica_groups = collective_inst->replica_groups();

      if (replica_groups.empty()) {
        return 0;
      }

      // Return the number of replica groups
      uint64_t num_groups = replica_groups.size();
      CHECK_GE(num_groups, 1)
          << "num_replica_groups must be at least 1, got: " << num_groups;
      return num_groups;
    }
  }

  return 0;
}

// Helper function to get replica group size from instruction
uint64_t GetReplicaGroupSize(const xla::HloInstruction *instr) {
  // Handle CollectivePermute separately since it doesn't inherit from
  // HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto *collective_permute =
        xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto &pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      uint64_t size = participants.size();
      CHECK_GE(size, 1) << "replica_group_size must be at least 1, got: "
                        << size;
      return size;
    }
    return 0;
  }

  // Handle other collective operations that inherit from
  // HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto *collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto &replica_groups = collective_inst->replica_groups();

      if (replica_groups.empty()) {
        return 0;
      }

      // Check symmetry: all groups should have the same size
      const size_t first_group_size = replica_groups[0].replica_ids_size();
      for (size_t i = 1; i < replica_groups.size(); ++i) {
        CHECK_EQ(replica_groups[i].replica_ids_size(), first_group_size)
            << "Expected symmetric replica groups for instruction: "
            << instr->ToString();
      }

      CHECK_GE(first_group_size, 1)
          << "replica_group_size must be at least 1, got: " << first_group_size;
      return first_group_size;
    }
  }

  return 0;
}

// Helper function to get data type from instruction result shape
std::string GetDataTypeFromInstruction(const xla::HloInstruction *instr) {
  if (!instr || !instr->shape().IsArray()) {
    return "unknown";
  }

  auto element_type = instr->shape().element_type();
  switch (element_type) {
  case xla::PrimitiveType::PRED:
    return "pred";
  case xla::PrimitiveType::S8:
    return "s8";
  case xla::PrimitiveType::S16:
    return "s16";
  case xla::PrimitiveType::S32:
    return "s32";
  case xla::PrimitiveType::S64:
    return "s64";
  case xla::PrimitiveType::U8:
    return "u8";
  case xla::PrimitiveType::U16:
    return "u16";
  case xla::PrimitiveType::U32:
    return "u32";
  case xla::PrimitiveType::U64:
    return "u64";
  case xla::PrimitiveType::F16:
    return "f16";
  case xla::PrimitiveType::F32:
    return "f32";
  case xla::PrimitiveType::F64:
    return "f64";
  case xla::PrimitiveType::BF16:
    return "bf16";
  case xla::PrimitiveType::C64:
    return "c64";
  case xla::PrimitiveType::C128:
    return "c128";
  case xla::PrimitiveType::F8E5M2:
    return "f8e5m2";
  case xla::PrimitiveType::F8E4M3FN:
    return "f8e4m3fn";
  case xla::PrimitiveType::F8E4M3B11FNUZ:
    return "f8e4m3b11fnuz";
  case xla::PrimitiveType::F8E5M2FNUZ:
    return "f8e5m2fnuz";
  case xla::PrimitiveType::F8E4M3FNUZ:
    return "f8e4m3fnuz";
  case xla::PrimitiveType::F4E2M1FN:
    return "f4e2m1fn";
  default:
    return "unknown";
  }
}

// Returns the datatype string used for compute/peak (for dots: first operand;
// otherwise result type). Use this for comp_stats "datatype" so FP4/FP8/BF16
// counts reflect the precision used for peak matrix ops.
std::string GetComputeDataTypeFromInstruction(const xla::HloInstruction *instr) {
  if (!instr || !instr->shape().IsArray()) {
    return "unknown";
  }
  xla::PrimitiveType element_type = instr->shape().element_type();
  if ((instr->opcode() == xla::HloOpcode::kDot ||
       instr->opcode() == xla::HloOpcode::kRaggedDot) &&
      instr->operand_count() >= 1 && instr->operand(0)->shape().IsArray()) {
    element_type = instr->operand(0)->shape().element_type();
  }
  switch (element_type) {
  case xla::PrimitiveType::PRED:
    return "pred";
  case xla::PrimitiveType::S8:
    return "s8";
  case xla::PrimitiveType::S16:
    return "s16";
  case xla::PrimitiveType::S32:
    return "s32";
  case xla::PrimitiveType::S64:
    return "s64";
  case xla::PrimitiveType::U8:
    return "u8";
  case xla::PrimitiveType::U16:
    return "u16";
  case xla::PrimitiveType::U32:
    return "u32";
  case xla::PrimitiveType::U64:
    return "u64";
  case xla::PrimitiveType::F16:
    return "f16";
  case xla::PrimitiveType::F32:
    return "f32";
  case xla::PrimitiveType::F64:
    return "f64";
  case xla::PrimitiveType::BF16:
    return "bf16";
  case xla::PrimitiveType::C64:
    return "c64";
  case xla::PrimitiveType::C128:
    return "c128";
  case xla::PrimitiveType::F8E5M2:
    return "f8e5m2";
  case xla::PrimitiveType::F8E4M3FN:
    return "f8e4m3fn";
  case xla::PrimitiveType::F8E4M3B11FNUZ:
    return "f8e4m3b11fnuz";
  case xla::PrimitiveType::F8E5M2FNUZ:
    return "f8e5m2fnuz";
  case xla::PrimitiveType::F8E4M3FNUZ:
    return "f8e4m3fnuz";
  case xla::PrimitiveType::F4E2M1FN:
    return "f4e2m1fn";
  default:
    return "unknown";
  }
}

// Helper function to add computation statistics to CSV data
void AddCompStatsToCSV(std::vector<std::vector<std::string>> &comp_stats_data,
                       int comp_id, int inst_count,
                       const std::string &deduplicated_name, double cost,
                       const xla::gpu::EstimateRunTimeData &runtime_data,
                       const std::string &device_name,
                       const xla::HloInstruction *instr,
                       bool is_entry, const std::string &hw_arch) {
  auto delimiter = '.';
  std::vector<std::string> parts = absl::StrSplit(deduplicated_name, delimiter);
  std::string group_name;
  if (parts.size() > 0) {
    group_name = parts.at(0);
  } else {
    group_name = "N/A";
  }

  // Calculate throughput: flops / exec_time (TFLOPs/s)
  double throughput = 0.0;
  if (runtime_data.exec_time > absl::ZeroDuration()) {
    // Convert exec_time to seconds for throughput calculation
    double exec_time_seconds = absl::ToDoubleSeconds(runtime_data.exec_time);
    throughput =
        (runtime_data.flops / exec_time_seconds) / 1e12; // Convert to TFLOPs/s
  }

  std::string op_name = std::string(instr->metadata().op_name());
  if (op_name.empty()) {
    op_name = std::string(instr->name());
  }

  comp_stats_data.push_back(
      {std::to_string(comp_id), std::to_string(inst_count),
       std::string(deduplicated_name), op_name, group_name, std::to_string(cost),
       std::to_string(runtime_data.flops / 1e12),        // tflops: FLOPs used for compute_time, in units of 10^12
       std::to_string(runtime_data.bytes_read / 1e9),    // Convert to GB
       std::to_string(runtime_data.bytes_written / 1e9), // Convert to GB
       absl::FormatDuration(runtime_data.compute_time),
       absl::FormatDuration(runtime_data.read_time),
       absl::FormatDuration(runtime_data.write_time),
       std::to_string(throughput),
       runtime_data.effective_matrix_tflops.has_value()
           ? std::to_string(*runtime_data.effective_matrix_tflops)
           : "",
       device_name,
       GetComputeDataTypeFromInstruction(instr),
       is_entry ? "true" : "false", hw_arch});
}

// Communication volume calculation with topology support
struct CommunicationVolume {
  double per_device_volume;
  double total_volume;
  double operand_size;
  double result_size;
  std::string pattern;
  std::string formula;
};

// Helper function to calculate ring-based communication volume
CommunicationVolume
CalculateRingVolume(double per_step_data, uint64_t replica_group_size,
                    const std::string &operation_name,
                    const std::string &formula, double multiplier = 1.0,
                    double operand_size = 0.0, double result_size = 0.0) {
  double total_steps = replica_group_size - 1;
  double per_device_volume = per_step_data * total_steps;
  double total_volume = per_device_volume * replica_group_size;

  return {per_device_volume * multiplier,
          total_volume * multiplier,
          operand_size,
          result_size,
          "Ring-based " + operation_name,
          formula};
}

CommunicationVolume
CalculateCommunicationVolume(const xla::HloInstruction *instr,
                             uint64_t replica_group_size) {

  double operand_size = static_cast<double>(
      xla::ShapeUtil::ByteSizeOf(instr->operand(0)->shape()));
  double result_size =
      static_cast<double>(xla::ShapeUtil::ByteSizeOf(instr->shape()));

  switch (instr->opcode()) {
  case xla::HloOpcode::kAllReduce: {
    CHECK_EQ(operand_size, result_size)
        << "AllReduce operand and result sizes must be equal. "
        << "Operand size: " << operand_size << ", Result size: " << result_size
        << " for instruction: " << instr->ToString();

    double per_step_data = operand_size / replica_group_size;
    std::string formula = "operand_size/" + std::to_string(replica_group_size) +
                          " * (" + std::to_string(replica_group_size) + "-1)";
    CommunicationVolume result =
        CalculateRingVolume(per_step_data, replica_group_size, "AllReduce",
                            formula, 2.0, operand_size, result_size);
    // When EP=1, this check fails.
    // CHECK_GT(result.per_device_volume, 0.0)
    //     << "per_device_comm_volume must be greater than 0.0, got: "
    //     << result.per_device_volume
    //     << instr->ToString();
    return result;
  }
  case xla::HloOpcode::kAllGather: {
    CHECK_GE(result_size, operand_size)
        << "AllGather result size must be greater than or equal to operand "
           "size. "
        << "Operand size: " << operand_size << ", Result size: " << result_size
        << " for instruction: " << instr->ToString();

    std::string formula =
        "operand_size * (" + std::to_string(replica_group_size) + "-1)";
    CommunicationVolume result =
        CalculateRingVolume(operand_size, replica_group_size, "AllGather",
                            formula, 1.0, operand_size, result_size);
    CHECK_GT(result.per_device_volume, 0.0)
        << "per_device_comm_volume must be greater than 0.0, got: "
        << result.per_device_volume;
    return result;
  }
  case xla::HloOpcode::kReduceScatter: {
    CHECK_LE(result_size, operand_size)
        << "ReduceScatter result size must be less than or equal to operand "
           "size. "
        << "Operand size: " << operand_size << ", Result size: " << result_size
        << " for instruction: " << instr->ToString();
    CommunicationVolume result;
    if (operand_size == result_size) {
      CHECK_EQ(replica_group_size, 1)
          << "In Reduce scatter, when operand_size == result_size, "
             "replica_group_size should be 1";
      result = {operand_size,
          operand_size * 2,
          operand_size,
          result_size,
          "Reduce-scatter with 1 replica group",
          "operand_size * 2"};
    } else {
    std::string formula =
        "result_size * (" + std::to_string(replica_group_size) + "-1)";
    result =
        CalculateRingVolume(result_size, replica_group_size, "ReduceScatter",
                            formula, 1.0, operand_size, result_size);
    }
    if (result.per_device_volume == 0.0) {
      // CommunicationVolume result = {operand_size,
      //   operand_size,
      //   operand_size,
      //   result_size,
      //   "Ring-based Local ReduceScatter",
      //   "operand_size * 1"};
      llvm::outs() << "WARN. Local reduce-scatter operation detected. "
                   << "Instruction: " << instr->ToString() << ", "
                   << "replica_group_size: " << replica_group_size << ", "
                   << "operand_size: " << operand_size << "\n";
      llvm::outs().flush();
    }
    CHECK_GE(result.per_device_volume, 0.0)
        << "per_device_comm_volume must be greater than 0.0, got: "
        << result.per_device_volume;
    return result;
  }
  case xla::HloOpcode::kCollectivePermute: {
    CHECK_EQ(operand_size, result_size)
        << "CollectivePermute operand and result sizes must be equal. "
        << "Operand size: " << operand_size << ", Result size: " << result_size
        << " for instruction: " << instr->ToString();

    CommunicationVolume result = {operand_size,
                                  operand_size * 2,
                                  operand_size,
                                  result_size,
                                  "Point-to-point permutation",
                                  "operand_size * 2"};
    CHECK_GT(result.per_device_volume, 0.0)
        << "per_device_comm_volume must be greater than 0.0, got: "
        << result.per_device_volume;
    return result;
  }
  default:
    return {0.0, 0.0, 0.0, 0.0, "Unknown", "Unknown"};
  }
}

namespace xla {
namespace gpu {

absl::Status RunAnalyticalLatencyCalculation(
    const AnalyticalLatencyCalculatorOpts& opts,
    const std::vector<int>& mesh_shape,
    const std::string& output_path_prefix) {
  CHECK(!opts.hlo_module_file.empty())
      << "Path to HLO module file required";
  CHECK(!opts.gpu_model_data_root.empty())
      << "gpu_model_data_root is required";
  CHECK(!opts.output_dir.empty()) << "output_dir is required";
  CHECK(!opts.mesh_shape.empty()) << "mesh_shape is required";

  const std::string specs_dir = ResolveSpecsDir(opts);
  const std::string configs_dir = ResolveConfigsDir(opts);
  absl::Status validation_status =
      ValidateHardwareArchitectures(opts.hardware_architectures, specs_dir);
  if (!validation_status.ok()) {
    return validation_status;
  }

  // Pipeline parallelism is Calcium-only. Reject up front when PP > 1 is
  // requested with any non-Calcium architecture; other archs assume PP=1.
  if (opts.num_pipeline_stages > 1) {
    for (const std::string& arch : opts.hardware_architectures) {
      std::string lower(arch);
      std::transform(lower.begin(), lower.end(), lower.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (lower.find("q250") == std::string::npos &&
          lower.find("calcium") == std::string::npos) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Pipeline parallelism (num_pipeline_stages=",
            opts.num_pipeline_stages,
            ") is only supported for q250/calcium. Got: ", arch,
            ". Other architectures assume PP=1; pass num_pipeline_stages=1."));
      }
    }
  }

  llvm::outs() << "Using hardware architectures: ";
  for (size_t i = 0; i < opts.hardware_architectures.size(); ++i) {
    if (i > 0)
      llvm::outs() << ", ";
    llvm::outs() << opts.hardware_architectures[i];
  }
  llvm::outs() << "\n";
  llvm::outs() << "Using output directory: " << opts.output_dir << "\n";
  llvm::outs() << "Using mesh shape: " << mesh_shape[0] << "x" << mesh_shape[1]
               << "x" << mesh_shape[2] << "\n";
  llvm::outs().flush();

  std::string output_path;
  if (tsl::io::IsAbsolutePath(opts.output_dir)) {
    output_path = opts.output_dir;
  } else if (!output_path_prefix.empty()) {
    output_path = tsl::io::JoinPath(output_path_prefix, opts.output_dir);
  } else {
    output_path = opts.output_dir;  // relative to current directory
  }
  if (!tsl::Env::Default()->FileExists(output_path).ok()) {
    absl::Status create_status =
        tsl::Env::Default()->RecursivelyCreateDir(output_path);
    if (!create_status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to create output directory: ", output_path, ": ",
          create_status.message()));
    }
  }
  CsvOut device_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "device_stats.csv"), true);
  CsvOut comp_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "comp_stats.csv"), false);
  CsvOut comm_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "comm_stats.csv"), true);
  CsvOut overlap_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "overlap_stats.csv"), true);
  CsvOut instruction_timeline_csv =
      createCsv(tsl::io::JoinPath(output_path, "instruction_timeline.csv"),
                true);
  if (!device_stats_csv.stream.is_open() || !comp_stats_csv.stream.is_open() ||
      !comm_stats_csv.stream.is_open() ||
      !overlap_stats_csv.stream.is_open() ||
      !instruction_timeline_csv.stream.is_open()) {
    return absl::InternalError("Failed to open one or more CSV output files");
  }

  std::vector<std::vector<std::string>> device_stats_data, comp_stats_data,
      comm_stats_data, overlap_stats_data, instruction_timeline_data,
      pipeline_stats_data;
  std::vector<std::string> device_stats_header = {
      "device_name",           "overlapped_latency_secs",
      "original_latency_secs", "compute_time_secs",
      "comm_time_secs",        "overlap_factor",
      "overlap_savings_secs",  "overlap_percentage",
      "total_instructions",    "hw_arch"};
  std::vector<std::string> comp_stats_header = {
      "comp_id",      "idx",       "inst",          "op_name",     "group",
      "latency(µs)",  "tflops",    "bytes_read_gb", "bytes_written_gb",
      "compute_time", "read_time", "write_time",    "throughput_tflops_per_sec",
      "effective_matrix_tflops",
      "device_name",  "datatype",  "is_entry",      "hw_arch"};
  std::vector<std::string> comm_stats_header = {"comp_id",
                                                "instruction_name",
                                                "opcode",
                                                "op_name",
                                                "device_name",
                                                "comm_type",
                                                "comm_cost_us",
                                                "replica_group_size",
                                                "num_replica_groups",
                                                "intranode_comm_vol_gb",
                                                "internode_comm_vol_gb",
                                                "intranode_comm_bw_gbps",
                                                "internode_comm_bw_gbps",
                                                "total_comm_vol_gb",
                                                "per_device_comm_vol_gb",
                                                "operand_size_bytes",
                                                "result_size_bytes",
                                                "datatype",
                                                "num_hops",
                                                "per_link_cost_us",
                                                "is_entry",
                                                "hw_arch"};
  std::vector<std::string> overlap_stats_header = {"device_name",
                                                   "overlap_factor",
                                                   "original_total_time_secs",
                                                   "overlapped_total_time_secs",
                                                   "total_compute_time_secs",
                                                   "total_comm_time_secs",
                                                   "overlap_savings_secs",
                                                   "overlap_percentage",
                                                   "total_instructions"};
  std::vector<std::string> instruction_timeline_header = {
      "device_name",
      "instruction_name",
      "instruction_type",
      "original_duration_us",
      "overlapped_start_time_us",
      "overlapped_end_time_us",
      "overlapped_duration_us",
      "is_compute",
      "is_comm"};

  device_stats_data.push_back(device_stats_header);
  comp_stats_data.push_back(comp_stats_header);
  comm_stats_data.push_back(comm_stats_header);
  overlap_stats_data.push_back(overlap_stats_header);
  instruction_timeline_data.push_back(instruction_timeline_header);

  // Pipeline-parallelism stats (Calcium-only). Header is appended only if a
  // PP run actually produces rows; the file is written only in that case.
  // Step 8b: t_handoff_us preserves the legacy column (now equals the
  // average per-boundary handoff). Three new columns at the end -
  // t_handoff_max_us, t_handoff_sum_us, pipeline_comm_overlap_factor -
  // expose the per-boundary aggregates that drive T_step / T_first
  // respectively. The column is named pipeline_comm_overlap_factor (not
  // just comm_overlap_factor) to disambiguate from the existing within-
  // stage compute/comm overlap_factor; the two operate at different layers.
  std::vector<std::string> pipeline_stats_header = {
      "device_name",
      "hw_arch",
      "num_pipeline_stages",
      "pipeline_microbatches",
      "pipeline_activation_bytes",
      "devices_per_stage",
      "t_stage_us",
      "t_handoff_us",
      "t_first_us",
      "t_step_us",
      "t_total_us",
      "bubble_time_us",
      "bubble_fraction",
      "bound_by",
      "t_handoff_max_us",
      "t_handoff_sum_us",
      "pipeline_comm_overlap_factor"};

  // Infer module format from file extension. .mlir is treated as HLO text (the
  // file content is a HLO variant, not MLIR/MHLO). Other extensions map to
  // their usual loader format.
  std::string format = std::string(tsl::io::Extension(opts.hlo_module_file));
  if (format == "mlir") {
    format = "hlo";
  }
  if (format.empty() ||
      (format != "mhlo" && format != "stablehlo" && format != "hlo" &&
       format != "txt" && format != "pb" && format != "pbtxt")) {
    format = "hlo";
  }

  std::unique_ptr<xla::HloModule> hlo_module =
      *xla::LoadModuleFromFile(opts.hlo_module_file, format, {});

  absl::flat_hash_map<
      std::string, std::pair<stream_executor::GpuDeviceInfoProto, std::string>>
      gpu_specs;
  for (const std::string &file_name : opts.hardware_architectures) {
    stream_executor::GpuTargetConfigProto proto;
    std::string spec_string;
    std::string path =
        tsl::io::JoinPath(specs_dir, absl::StrCat(file_name, ".txtpb"));
    absl::Status is_file_read =
        tsl::ReadFileToString(tsl::Env::Default(), path, &spec_string);
    if (is_file_read.ok()) {
      tsl::protobuf::TextFormat::ParseFromString(spec_string, &proto);
    } else {
      return absl::NotFoundError(
          absl::StrCat("Device spec file read failed for: ", path));
    }
    llvm::outs().flush();
    gpu_specs[proto.device_description_str()] =
        std::make_pair(proto.gpu_device_info(), file_name);
  }

  auto const SEPARATOR = ",";
  int64_t pointer_size = 8;
  for (const auto &pair : gpu_specs) {
    auto device_name = pair.first;
    stream_executor::GpuDeviceInfoProto gpu_device_info_pb = pair.second.first;
    std::string spec_file_name = pair.second.second;

    // Optional: scale memory bandwidth to verify memory-bound behavior.
    if (opts.scale_memory_bandwidth > 0.0 && opts.scale_memory_bandwidth != 1.0) {
      int64_t bw = gpu_device_info_pb.memory_bandwidth();
      gpu_device_info_pb.set_memory_bandwidth(static_cast<int64_t>(
          static_cast<double>(bw) * opts.scale_memory_bandwidth));
      llvm::outs() << "Scaled memory bandwidth by " << opts.scale_memory_bandwidth
                   << "x (effective: "
                   << (static_cast<double>(bw) * opts.scale_memory_bandwidth / 1e12)
                   << " TB/s)\n";
    }

    absl::StatusOr<stream_executor::DeviceDescription> gpu_device_info_result =
        stream_executor::DeviceDescription::FromProto(gpu_device_info_pb);

    if (!gpu_device_info_result.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to create DeviceDescription: ", device_name));
    }

    stream_executor::DeviceDescription gpu_device_info =
        gpu_device_info_result.value();
    gpu_device_info.set_name(device_name);
    const uint64_t memory_limit =
        hlo_module->config().device_memory_size() != 0
            ? hlo_module->config().device_memory_size()
            : gpu_device_info.device_memory_size() * 80 / 100;

    const auto num_partitions = hlo_module->config().num_partitions();
    auto collective_overlap_limit = 1;
    xla::SchedulerConfig scheduler_config = xla::gpu::MakeGPUSchedulerConfig(
        memory_limit * num_partitions, collective_overlap_limit);

    auto gle_latency_estimator =
        std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);
    mlir::MLIRContext mlir_context;
    xla::gpu::GpuPerformanceModelOwning gpu_performance_model_(gpu_device_info, &mlir_context);

    if (opts.fix_ragged_dot_flops) {
      for (xla::HloComputation* computation : hlo_module->computations()) {
        std::vector<xla::HloInstruction*> instructions_to_replace;
        
        for (xla::HloInstruction* instr : computation->instructions()) {
            if (instr->opcode() == xla::HloOpcode::kDot) {
            std::string op_name = instr->metadata().op_name();
            if (absl::StrContains(op_name, "ragged_dot")) {
              const xla::Shape& original_lhs_shape = instr->operand(0)->shape();
              const xla::Shape& original_rhs_shape = instr->operand(1)->shape();
              
              bool shapes_valid = (original_lhs_shape.dimensions().size() == 3 &&
                                  original_rhs_shape.dimensions().size() == 3 &&
                                  original_lhs_shape.dimensions(0) == original_rhs_shape.dimensions(0));
              
              if (shapes_valid) {
                instructions_to_replace.push_back(instr);
              }
            }
          }
        }
        
        for (xla::HloInstruction* dot_instr : instructions_to_replace) {
          const xla::Shape& original_lhs_shape = dot_instr->operand(0)->shape();
          const xla::Shape& original_rhs_shape = dot_instr->operand(1)->shape();
          int64_t overflow_factor = 2;
          int64_t g_dim = original_lhs_shape.dimensions(0);
          int64_t m_dim = original_lhs_shape.dimensions(1);
          int64_t k_dim = original_lhs_shape.dimensions(2);
          int64_t n_dim = original_rhs_shape.dimensions(2);
          
          // Divide first dimension by overflow_factor
          int64_t m_dim_adjusted = m_dim / overflow_factor;
          
          xla::Shape lhs_shape_for_ragged = xla::ShapeUtil::DeleteDimension(0, original_lhs_shape);
          // Adjust the first dimension (m_dim) by dividing by overflow_factor
          lhs_shape_for_ragged.set_dimensions(0, m_dim_adjusted);
          
          std::vector<int64_t> start_indices = {0, 0, 0};
          std::vector<int64_t> limit_indices = {1, m_dim_adjusted, k_dim};
          std::vector<int64_t> strides = {1, 1, 1};
          
          xla::Shape sliced_lhs_shape = xla::ShapeUtil::MakeShape(
              original_lhs_shape.element_type(), {1, m_dim_adjusted, k_dim});
          xla::HloInstruction* sliced_lhs = computation->AddInstruction(
              xla::HloInstruction::CreateSlice(
                  sliced_lhs_shape,
                  dot_instr->mutable_operand(0),
                  start_indices,
                  limit_indices,
                  strides));
          
          xla::HloInstruction* reshaped_lhs = computation->AddInstruction(
              xla::HloInstruction::CreateReshape(lhs_shape_for_ragged, sliced_lhs));
          
          xla::DotDimensionNumbers dot_dnums;
          dot_dnums.add_lhs_contracting_dimensions(1);
          dot_dnums.add_rhs_contracting_dimensions(1);
          
          xla::RaggedDotDimensionNumbers ragged_dnums;
          *ragged_dnums.mutable_dot_dimension_numbers() = dot_dnums;
          ragged_dnums.add_lhs_ragged_dimensions(0);
          
          xla::Shape group_sizes_shape = xla::ShapeUtil::MakeShape(
              xla::PrimitiveType::S32, {g_dim});
          
          xla::Literal group_sizes_literal = xla::LiteralUtil::CreateFullWithDescendingLayout(
              group_sizes_shape.dimensions(), static_cast<int32_t>(1));
          xla::HloInstruction* group_sizes = computation->AddInstruction(
              xla::HloInstruction::CreateConstant(std::move(group_sizes_literal)));
          
          // Adjust output shape: first dimension divided by overflow_factor
          absl::StatusOr<xla::Shape> ragged_output_shape =
              xla::ShapeUtil::MakeShape(original_rhs_shape.element_type(), {m_dim_adjusted, n_dim});
          
          if (ragged_output_shape.ok()) {
            xla::HloInstruction* new_ragged_dot = computation->AddInstruction(
                xla::HloInstruction::CreateRaggedDot(
                    ragged_output_shape.value(), 
                    reshaped_lhs,
                    dot_instr->mutable_operand(1),
                    group_sizes,
                    ragged_dnums, 
                    dot_instr->precision_config()));
            
            new_ragged_dot->set_metadata(dot_instr->metadata());
            
            // Use ReplaceInstructionWithDifferentShape since output shape changed
            // (m_dim_adjusted instead of m_dim)
            TF_CHECK_OK(computation->ReplaceInstructionWithDifferentShape(
                dot_instr, new_ragged_dot));
          }
        }
      }
    }
    
    if (opts.dump_modified_module && !opts.output_dir.empty()) {
      std::string hlo_output_path =
          tsl::io::JoinPath(output_path, "modified_module.hlo");
      std::ofstream hlo_file(hlo_output_path);
      if (hlo_file.is_open()) {
        xla::HloPrintOptions print_options;
        print_options.set_print_large_constants(true);
        print_options.set_print_backend_config(true);
        print_options.set_include_layout_in_shapes(true);
        hlo_file << hlo_module->ToString(print_options);
        hlo_file.close();
      }
    }
    
    auto ale_cost_analysis = std::make_unique<xla::gpu::GpuHloCostAnalysis>(
        xla::gpu::GpuHloCostAnalysis::Options{
            xla::gpu::ShapeSizeBytesFunction(pointer_size),
            /*per_second_rates=*/{},
            /*min_latencies_seconds=*/{},
            /*count_multiple_input_accesses=*/true});
    auto coll_cost_analysis = std::make_unique<xla::gpu::GpuHloCostAnalysis>(
        xla::gpu::GpuHloCostAnalysis::Options{
            xla::gpu::ShapeSizeBytesFunction(pointer_size),
            /*per_second_rates=*/{},
            /*min_latencies_seconds=*/{},
            /*count_multiple_input_accesses=*/true});
    
    for (xla::HloComputation* computation : hlo_module->computations()) {
      TF_CHECK_OK(computation->Accept(ale_cost_analysis.get()));
      TF_CHECK_OK(computation->Accept(coll_cost_analysis.get()));
    }

    auto gpu_latency_estimator =
        std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);

    auto inst_count = 0;
    auto comp_count = 0;
    // Total cost (compute + communication) for current computation
    auto comp_total_cost = 0.0;
    // Track compute time for entry computation only
    auto entry_compute_time = 0.0;
    // Track communication time for entry computation only
    auto entry_comm_time = 0.0;
    absl::flat_hash_map<const xla::HloComputation *, double> computation_map;
    absl::flat_hash_map<const xla::HloComputation *, int> computation_idx;
    // Track timeline entries per computation (for while loop body/condition reuse)
    absl::flat_hash_map<const xla::HloComputation *,
                        std::vector<InstructionTimelineEntry>>
        computation_timeline_map;
    std::vector<InstructionTimelineEntry> instruction_timeline;
    // Track which computations are used during entry computation cost calculation
    absl::flat_hash_set<const xla::HloComputation *> used_computations;
    auto cluster_config = GetClusterConfigByName(spec_file_name, configs_dir);
    if (!cluster_config) {
      llvm::outs() << "Error: Failed to get cluster config for device: "
                   << spec_file_name << "\n";
      llvm::outs().flush();
      continue;
    }
    // Step 9: per-call selection of the Calcium hierarchical AllReduce
    // cost model. -1 (default) keeps the built-in OFF state for byte-
    // stable flat AR; 0/1 explicitly turn the model off/on. There is
    // intentionally no config-file key for this; the FFI / CLI is the
    // only way to enable hierarchical AR. Silently ignored for non-
    // Calcium archs.
    if (opts.hierarchical_allreduce_enabled >= 0) {
      if (auto* calcium =
              dynamic_cast<CalciumClusterConfig*>(cluster_config.get())) {
        calcium->SetHierarchicalAllReduceEnabled(
            opts.hierarchical_allreduce_enabled != 0);
      }
    }
    // First pass: Calculate costs for all computations and populate computation_map
    for (xla::HloComputation *computation : hlo_module->computations()) {
      comp_total_cost = 0;
      auto is_entry_comp = computation->IsEntryComputation();
      // Build timeline for this computation
      // Use MakeInstructionPostOrder() to get instructions in execution order
      // (topological order), which preserves the logical order even after
      // instruction replacements (e.g., ragged dot replacements).
      std::vector<InstructionTimelineEntry> comp_timeline;
      std::vector<xla::HloInstruction*> instruction_order = computation->MakeInstructionPostOrder();
      for (xla::HloInstruction *instr : instruction_order) {
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
        xla::gpu::EstimateRunTimeData runtime_data{};
        switch (opcode) {
        case xla::HloOpcode::kWhile: {
          auto comps = {instr->while_body(), instr->while_condition()};
          auto while_cost = 0.0;
          for (auto comp : comps) {
            auto it = computation_map.find(comp);
            auto it2 = computation_idx.find(comp);
            //llvm::outs() << "Adding cost of comp:" << it2->second << "\n";
            CHECK(it != computation_map.end());
            while_cost += it->second;
            // Usage tracking is now done in a separate pass after all costs are calculated
          }
          
          // Try to get trip count from backend config, or calculate it dynamically
          int64_t trip_count = 0;
          absl::StatusOr<xla::WhileLoopBackendConfig> while_config =
              instr->backend_config<xla::WhileLoopBackendConfig>();
          if (while_config.ok() && while_config->has_known_trip_count()) {
            trip_count = while_config->known_trip_count().n();
          } else {
            // Fallback: Try to compute trip count dynamically
            auto computed_trip_count = xla::ComputeWhileLoopTripCount(instr);
            if (computed_trip_count.has_value()) {
              trip_count = *computed_trip_count;
            } else {
              return absl::InternalError(absl::StrCat(
                  "Could not determine trip count for while loop: ",
                  instr->name()));
            }
          }
          
          cost = while_cost * trip_count;
          
          // Get timeline entries from body and condition computations
          auto body_timeline_it = computation_timeline_map.find(instr->while_body());
          auto cond_timeline_it =
              computation_timeline_map.find(instr->while_condition());
          
          CHECK(body_timeline_it != computation_timeline_map.end())
              << "While loop body timeline not found";
          CHECK(cond_timeline_it != computation_timeline_map.end())
              << "While loop condition timeline not found";
          
          const auto &body_timeline = body_timeline_it->second;
          const auto &cond_timeline = cond_timeline_it->second;
          
          // Repeat body and condition timelines for each iteration
          for (int64_t iter = 0; iter < trip_count; ++iter) {
            // Add condition check timeline entries for this iteration
            for (const auto &cond_entry : cond_timeline) {
              InstructionTimelineEntry iter_cond_entry = cond_entry;
              iter_cond_entry.instruction_name =
                  absl::StrCat(cond_entry.instruction_name, "_iter_", iter);
              comp_timeline.push_back(iter_cond_entry);
            }
            
            // Add body timeline entries for this iteration
            for (const auto &body_entry : body_timeline) {
              InstructionTimelineEntry iter_body_entry = body_entry;
              iter_body_entry.instruction_name =
                  absl::StrCat(body_entry.instruction_name, "_iter_", iter);
              comp_timeline.push_back(iter_body_entry);
            }
          }
          
          if (is_entry_comp) {
            entry_compute_time += cost;
          }
          runtime_data =
              gpu_performance_model_.Get().EstimateRunTimeForInstruction(
                  instr, &*ale_cost_analysis);

          // Add computation statistics to CSV data for all computations
          if (cost > 0) {
            inst_count += 1;
            AddCompStatsToCSV(comp_stats_data, comp_count, inst_count,
                              std::string(deduplicated_name), cost,
                              runtime_data, device_name, instr, is_entry_comp,
                              spec_file_name);
          }
          break;
        }
        case xla::HloOpcode::kReduceScatter:
        case xla::HloOpcode::kCollectivePermute:
        case xla::HloOpcode::kAllGather:
        case xla::HloOpcode::kAllReduce: {

          // Calculate communication cost statistics
          uint64_t replica_group_size = GetReplicaGroupSize(instr);
          uint64_t num_replica_groups = GetNumReplicaGroups(instr);
          CommunicationVolume comm_volume =
              CalculateCommunicationVolume(instr, replica_group_size);
          CommCostStats comm_stats;
          // Get device IDs and mesh shape for TPU communication cost
          // calculation
          auto device_ids = GetDeviceIdsFromOneReplicaGroup(instr);
          // Use mesh shape from command line argument
          comm_stats = cluster_config->CalculateCommCost(
              comm_volume.per_device_volume, gpu_device_info, instr,
              replica_group_size, num_replica_groups, device_ids, mesh_shape,
              spec_file_name, "");
          cost = comm_stats.comm_cost_us;

          if (is_entry_comp) {
            entry_comm_time += cost;
          }

          // Get op_name from metadata
          std::string op_name = std::string(instr->metadata().op_name());
          if (op_name.empty()) {
            op_name = std::string(instr->name());
          }

          std::vector<std::string> comm_row = {
              std::to_string(comp_count),
              std::string(deduplicated_name),
              std::string(xla::HloOpcodeString(opcode)),
              op_name,
              device_name,
              CommTypeToString(comm_stats.comm_type),
              std::to_string(comm_stats.comm_cost_us),
              std::to_string(replica_group_size),
              std::to_string(num_replica_groups),
              std::to_string(comm_stats.intranode_comm_vol_gb),
              std::to_string(comm_stats.internode_comm_vol_gb),
              std::to_string(comm_stats.intranode_comm_bw_gbps),
              std::to_string(comm_stats.internode_comm_bw_gbps),
              std::to_string(comm_stats.total_comm_vol_gb),
              std::to_string(comm_stats.per_device_comm_vol_gb),
              std::to_string(comm_volume.operand_size),
              std::to_string(comm_volume.result_size),
              GetDataTypeFromInstruction(instr),
              std::to_string(comm_stats.num_hops),
              std::to_string(comm_stats.per_link_cost_us),
              is_entry_comp ? "true" : "false",
              spec_file_name};
          comm_stats_data.push_back(comm_row);

          if (cost > 0) {
            inst_count += 1;
          }
          break;
        }
        case xla::HloOpcode::kDot:
        case xla::HloOpcode::kRaggedDot: {
          runtime_data =
              gpu_performance_model_.Get().EstimateRunTimeForInstruction(
                  instr, &*ale_cost_analysis);
          cost = absl::ToDoubleMicroseconds(runtime_data.exec_time);

          std::string op_name_check = std::string(instr->metadata().op_name());
          bool is_ragged_dot_op = absl::StrContains(op_name_check, "ragged_dot");

          if (is_entry_comp) {
            entry_compute_time += cost;
          }
          if (cost > 0) {
            inst_count += 1;
            AddCompStatsToCSV(comp_stats_data, comp_count, inst_count,
                              std::string(deduplicated_name), cost,
                              runtime_data, device_name, instr, is_entry_comp,
                              spec_file_name);
          }
          break;
        }
        case xla::HloOpcode::kConditional: {
          double max_branch_cost = 0.0;
          
          for (int32_t branch_idx = 0; branch_idx < instr->branch_count();
               ++branch_idx) {
            auto *branch_comp = instr->branch_computation(branch_idx);
            auto branch_cost_it = computation_map.find(branch_comp);
            
            if (branch_cost_it != computation_map.end()) {
              double branch_cost = branch_cost_it->second;
              max_branch_cost = std::max(max_branch_cost, branch_cost);
              
              auto branch_timeline_it = computation_timeline_map.find(branch_comp);
              if (branch_timeline_it != computation_timeline_map.end()) {
                const auto &branch_timeline = branch_timeline_it->second;
                
                for (const auto &branch_entry : branch_timeline) {
                  InstructionTimelineEntry conditional_entry = branch_entry;
                  conditional_entry.instruction_name = absl::StrCat(
                      branch_entry.instruction_name, "_branch_", branch_idx);
                  comp_timeline.push_back(conditional_entry);
                }
              }
            }
          }
          
          cost = max_branch_cost;
          
          if (is_entry_comp) {
            entry_compute_time += cost;
          }
          
          runtime_data =
              gpu_performance_model_.Get().EstimateRunTimeForInstruction(
                  instr, &*ale_cost_analysis);
          
          if (cost > 0) {
            inst_count += 1;
            AddCompStatsToCSV(comp_stats_data, comp_count, inst_count,
                              std::string(deduplicated_name), cost,
                              runtime_data, device_name, instr, is_entry_comp,
                              spec_file_name);
          }
          break;
        }
        default: {
          cost = 0.0;
          break;
        }
        };

        if (cost > 0.0 && opcode != xla::HloOpcode::kWhile &&
            opcode != xla::HloOpcode::kConditional) {
          InstructionTimelineEntry timeline_entry;
          timeline_entry.instruction = instr;
          timeline_entry.duration_us = cost;
          timeline_entry.start_time_us = 0.0;
          timeline_entry.end_time_us = 0.0;
          timeline_entry.instruction_name = std::string(deduplicated_name);

          switch (opcode) {
          case xla::HloOpcode::kReduceScatter:
          case xla::HloOpcode::kCollectivePermute:
          case xla::HloOpcode::kAllGather:
          case xla::HloOpcode::kAllReduce: {
            timeline_entry.is_compute_instruction = false;
            timeline_entry.is_comm_instruction = true;
            break;
          }
          case xla::HloOpcode::kDot:
          case xla::HloOpcode::kRaggedDot: {
            timeline_entry.is_compute_instruction = true;
            timeline_entry.is_comm_instruction = false;
            break;
          }
          default: {
            timeline_entry.is_compute_instruction = false;
            timeline_entry.is_comm_instruction = false;
            break;
          }
          }

          comp_timeline.push_back(timeline_entry);
        }

        comp_total_cost += cost;
      }

      computation_map.insert({computation, comp_total_cost});
      computation_idx.insert({computation, comp_count});
      computation_timeline_map.insert({computation, comp_timeline});
      instruction_timeline.insert(instruction_timeline.end(), comp_timeline.begin(),
                                  comp_timeline.end());
      comp_count += 1;
    }
    
    auto *entry_computation = hlo_module->entry_computation();
    if (entry_computation) {
      std::function<void(const xla::HloComputation *)> mark_used =
          [&](const xla::HloComputation *comp) {
            if (used_computations.find(comp) != used_computations.end()) {
              return; // Already marked
            }
            used_computations.insert(comp);
            
            // Mark all computations used by this computation
            for (xla::HloInstruction *instr : comp->instructions()) {
              if (instr->opcode() == xla::HloOpcode::kWhile) {
                mark_used(instr->while_body());
                mark_used(instr->while_condition());
              } else if (instr->opcode() == xla::HloOpcode::kConditional) {
                for (int32_t branch_idx = 0; branch_idx < instr->branch_count();
                     ++branch_idx) {
                  mark_used(instr->branch_computation(branch_idx));
                }
              } else {
                for (xla::HloComputation *called_comp :
                     instr->called_computations()) {
                  mark_used(called_comp);
                }
              }
            }
          };
      mark_used(entry_computation);
    }

    // Validation: Check if all computations with non-zero cost were used in entry computation
    {
      double entry_cost = 0.0;
      auto entry_it = computation_map.find(entry_computation);
      if (entry_it != computation_map.end()) {
        entry_cost = entry_it->second;
      }
      
      // Find computations with cost that were not used
      std::vector<const xla::HloComputation *> unused_with_cost;
      for (const auto &pair : computation_map) {
        if (pair.second > 0.0 && 
            pair.first != entry_computation &&
            used_computations.find(pair.first) == used_computations.end()) {
          unused_with_cost.push_back(pair.first);
        }
      }
      
      llvm::outs() << "\n=== Entry Computation Cost Validation ===\n";
      llvm::outs() << "Entry computation cost: " << entry_cost / 1e6 << " seconds\n";
      llvm::outs() << "Computations used in entry cost calculation: " 
                   << used_computations.size() << "\n";
      
      if (!unused_with_cost.empty()) {
        llvm::errs() << "ERROR: " << unused_with_cost.size()
                     << " computations with non-zero cost were NOT used in entry computation:\n";
        double unused_cost_total = 0.0;
        for (const auto *comp : unused_with_cost) {
          double comp_cost = computation_map.find(comp)->second;
          unused_cost_total += comp_cost;
          llvm::errs() << "  - " << comp->name() << " (cost: " 
                       << comp_cost / 1e6 << "s)\n";
        }
        llvm::errs() << "Total unused cost: " << unused_cost_total / 1e6 << " seconds\n";
        llvm::errs() << "This indicates that some sub-computation costs are missing from entry cost.\n";
        llvm::outs() << "Validation: FAILED\n";
      } else {
        llvm::outs() << "Validation: PASSED\n";
        llvm::outs() << "All computations with non-zero cost were used in entry computation.\n";
      }
      
      llvm::outs() << "==========================================\n";
      llvm::outs().flush();
    }

    // Calculate instruction-level overlapped latency
    OverlapStats overlap_stats = CalculateInstructionLevelOverlap(
        instruction_timeline, opts.overlap_factor);

    llvm::outs() << "=== Entry Computation Timing Breakdown ===\n";
    llvm::outs() << "Device: " << device_name << "\n";
    llvm::outs() << "Total Instructions Processed: "
                 << instruction_timeline.size() << "\n";
    if (opts.overlap_factor > 0.0) {
      llvm::outs() << "Overlap Factor: " << opts.overlap_factor << "\n";
      llvm::outs() << "Original Total Time: "
                   << overlap_stats.original_total_time_us / 1e6
                   << " seconds\n";
      llvm::outs() << "Overlapped Total Time: "
                   << overlap_stats.overlapped_total_time_us / 1e6
                   << " seconds\n";
      llvm::outs() << "Overlap Savings: "
                   << overlap_stats.overlap_savings_us / 1e6 << " seconds ("
                   << overlap_stats.overlap_percentage << "%)\n";
    } else {
      llvm::outs() << "Total Compute Time: "
                   << overlap_stats.total_compute_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Total Communication Time: "
                   << overlap_stats.total_comm_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Total Time: "
                   << overlap_stats.original_total_time_us / 1e6
                   << " seconds\n";
    }
    llvm::outs() << "==========================================\n";
    llvm::outs().flush();

    // Use overlapped time for device_stats
    auto overlapped_comp_cost_in_secs =
        overlap_stats.overlapped_total_time_us / 1e6;
    auto original_comp_cost_in_secs =
        overlap_stats.original_total_time_us / 1e6;
    auto total_compute_time_in_secs = overlap_stats.total_compute_time_us / 1e6;
    auto total_comm_time_in_secs = overlap_stats.total_comm_time_us / 1e6;
    device_stats_data.push_back(
        {device_name, std::to_string(overlapped_comp_cost_in_secs),
         std::to_string(original_comp_cost_in_secs),
         std::to_string(total_compute_time_in_secs),
         std::to_string(total_comm_time_in_secs),
         std::to_string(opts.overlap_factor),
         std::to_string(overlap_stats.overlap_savings_us / 1e6),
         std::to_string(overlap_stats.overlap_percentage),
         std::to_string(instruction_timeline.size()),
         spec_file_name});

    // Add overlap statistics to CSV data
    std::vector<std::string> overlap_row = {
        device_name,
        std::to_string(opts.overlap_factor),
        std::to_string(overlap_stats.original_total_time_us / 1e6),
        std::to_string(overlap_stats.overlapped_total_time_us / 1e6),
        std::to_string(overlap_stats.total_compute_time_us / 1e6),
        std::to_string(overlap_stats.total_comm_time_us / 1e6),
        std::to_string(overlap_stats.overlap_savings_us / 1e6),
        std::to_string(overlap_stats.overlap_percentage),
        std::to_string(instruction_timeline.size())};
    overlap_stats_data.push_back(overlap_row);

    // Pipeline-parallelism modeling (Calcium-only).
    // The FFI layer rejects PP > 1 for non-Calcium archs; here we still guard
    // with a dynamic_cast to avoid corrupting CSV output if some future caller
    // bypasses validation.
    if (opts.num_pipeline_stages > 1) {
      CalciumClusterConfig* calcium =
          dynamic_cast<CalciumClusterConfig*>(cluster_config.get());
      if (calcium != nullptr) {
        const int64_t activation_bytes =
            (opts.pipeline_activation_bytes > 0)
                ? opts.pipeline_activation_bytes
                : InferActivationBytes(*hlo_module);
        // devices_per_stage = product(mesh_shape) per the agreed convention.
        int devices_per_stage = 1;
        for (int d : mesh_shape) {
          if (d > 0) devices_per_stage *= d;
        }

        const double t_stage_us = overlap_stats.original_total_time_us;
        const PipelineHandoffStats handoffs =
            calcium->CalculatePipelineHandoffCosts(
                activation_bytes, devices_per_stage,
                opts.num_pipeline_stages,
                opts.pipeline_comm_overlap_factor);

        PipelineMetrics pm = ComputePipelineMetrics(
            opts.num_pipeline_stages, activation_bytes,
            opts.pipeline_microbatches, devices_per_stage, t_stage_us,
            handoffs);

        pipeline_stats_data.push_back({
            device_name,
            spec_file_name,
            std::to_string(pm.num_pipeline_stages),
            std::to_string(pm.pipeline_microbatches),
            std::to_string(pm.pipeline_activation_bytes),
            std::to_string(pm.devices_per_stage),
            std::to_string(pm.t_stage_us),
            std::to_string(pm.t_handoff_us),
            std::to_string(pm.t_first_us),
            std::to_string(pm.t_step_us),
            std::to_string(pm.t_total_us),
            std::to_string(pm.bubble_time_us),
            std::to_string(pm.bubble_fraction),
            pm.bound_by,
            std::to_string(pm.t_handoff_max_us),
            std::to_string(pm.t_handoff_sum_us),
            std::to_string(pm.pipeline_comm_overlap_factor),
        });

        llvm::outs() << "=== Pipeline Parallelism (Calcium) ===\n";
        llvm::outs() << "  num_pipeline_stages: " << pm.num_pipeline_stages
                     << "\n";
        llvm::outs() << "  pipeline_microbatches: "
                     << pm.pipeline_microbatches << "\n";
        llvm::outs() << "  pipeline_activation_bytes: "
                     << pm.pipeline_activation_bytes << "\n";
        llvm::outs() << "  devices_per_stage: " << pm.devices_per_stage
                     << "\n";
        llvm::outs() << "  t_stage_us:        " << pm.t_stage_us << "\n";
        llvm::outs() << "  t_handoff_avg_us:  " << pm.t_handoff_us << "\n";
        llvm::outs() << "  t_handoff_max_us:  " << pm.t_handoff_max_us << "\n";
        llvm::outs() << "  t_handoff_sum_us:  " << pm.t_handoff_sum_us << "\n";
        llvm::outs() << "  pipeline_comm_overlap_factor: "
                     << pm.pipeline_comm_overlap_factor << "\n";
        llvm::outs() << "  t_first_us:        " << pm.t_first_us << "\n";
        llvm::outs() << "  t_step_us:         " << pm.t_step_us << "\n";
        llvm::outs() << "  t_total_us:        " << pm.t_total_us << "\n";
        llvm::outs() << "  bubble_time_us:    " << pm.bubble_time_us << "\n";
        llvm::outs() << "  bubble_fraction:   " << pm.bubble_fraction << "\n";
        llvm::outs() << "  bound_by:          " << pm.bound_by << "\n";
        llvm::outs() << "======================================\n";
        llvm::outs().flush();
      }
    }

    // Add instruction timeline to CSV data
    for (const auto &entry : overlap_stats.timeline) {
      std::string instruction_type = "other";
      if (entry.is_compute_instruction)
        instruction_type = "compute";
      if (entry.is_comm_instruction)
        instruction_type = "comm";

      std::vector<std::string> timeline_row = {
          device_name,
          entry.instruction_name,
          instruction_type,
          std::to_string(entry.duration_us),
          std::to_string(entry.start_time_us),
          std::to_string(entry.end_time_us),
          std::to_string(entry.end_time_us - entry.start_time_us),
          entry.is_compute_instruction ? "true" : "false",
          entry.is_comm_instruction ? "true" : "false"};
      instruction_timeline_data.push_back(timeline_row);
    }

    llvm::outs().flush();
  }
  writeCsv(device_stats_csv.stream, device_stats_data,
           device_stats_csv.write_header);
  writeCsv(comp_stats_csv.stream, comp_stats_data,
           comp_stats_csv.write_header);

  // Count compute ops by datatype (and hw_arch) for summary and CSV.
  constexpr size_t kCompStatsDatatypeCol = 15;
  constexpr size_t kCompStatsHwArchCol = 17;
  std::map<std::string, std::map<std::string, int>> compute_op_counts;
  for (size_t i = 1; i < comp_stats_data.size(); ++i) {
    const auto &row = comp_stats_data[i];
    if (row.size() > kCompStatsHwArchCol) {
      compute_op_counts[row[kCompStatsHwArchCol]][row[kCompStatsDatatypeCol]]++;
    }
  }
  CsvOut compute_op_counts_csv =
      createCsv(tsl::io::JoinPath(output_path, "compute_op_counts.csv"), true);
  if (compute_op_counts_csv.stream.is_open()) {
    std::vector<std::vector<std::string>> count_data = {
        {"hw_arch", "datatype", "count"}};
    for (const auto &arch_pair : compute_op_counts) {
      const std::string &hw_arch = arch_pair.first;
      std::vector<std::string> summary_parts;
      for (const auto &dt_pair : arch_pair.second) {
        count_data.push_back(
            {hw_arch, dt_pair.first, std::to_string(dt_pair.second)});
        summary_parts.push_back(
            absl::StrCat(dt_pair.first, "=", dt_pair.second));
      }
      llvm::outs() << "Compute op counts (" << hw_arch
                   << "): " << absl::StrJoin(summary_parts, ", ") << "\n";
    }
    writeCsv(compute_op_counts_csv.stream, count_data,
             compute_op_counts_csv.write_header);
    compute_op_counts_csv.stream.close();
  }

  // Count compute-bound vs memory-bound ops from comp_stats (compute_time vs read+write).
  constexpr size_t kCompStatsComputeTimeCol = 9;
  constexpr size_t kCompStatsReadTimeCol = 10;
  constexpr size_t kCompStatsWriteTimeCol = 11;
  std::map<std::string, std::pair<int, int>> compute_bound_counts;  // hw_arch -> (compute_bound, memory_bound)
  for (size_t i = 1; i < comp_stats_data.size(); ++i) {
    const auto &row = comp_stats_data[i];
    if (row.size() <= kCompStatsHwArchCol) continue;
    double compute_sec =
        ParseDurationToSeconds(row[kCompStatsComputeTimeCol]);
    double read_sec = ParseDurationToSeconds(row[kCompStatsReadTimeCol]);
    double write_sec = ParseDurationToSeconds(row[kCompStatsWriteTimeCol]);
    std::string hw = row[kCompStatsHwArchCol];
    if (compute_sec >= read_sec + write_sec) {
      compute_bound_counts[hw].first++;
    } else {
      compute_bound_counts[hw].second++;
    }
  }
  for (const auto &arch_pair : compute_bound_counts) {
    const std::string &hw_arch = arch_pair.first;
    int compute_bound = arch_pair.second.first;
    int memory_bound = arch_pair.second.second;
    llvm::outs() << "Compute-bound ops (" << hw_arch << "): " << compute_bound
                 << ", Memory-bound ops (" << hw_arch << "): " << memory_bound
                 << "\n";
  }
  CsvOut bound_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "compute_memory_bound_stats.csv"),
                true);
  if (bound_stats_csv.stream.is_open()) {
    std::vector<std::vector<std::string>> bound_data = {
        {"hw_arch", "compute_bound", "memory_bound"}};
    for (const auto &arch_pair : compute_bound_counts) {
      bound_data.push_back(
          {arch_pair.first, std::to_string(arch_pair.second.first),
           std::to_string(arch_pair.second.second)});
    }
    writeCsv(bound_stats_csv.stream, bound_data, bound_stats_csv.write_header);
    bound_stats_csv.stream.close();
  }

  writeCsv(comm_stats_csv.stream, comm_stats_data,
           comm_stats_csv.write_header);
  writeCsv(overlap_stats_csv.stream, overlap_stats_data,
           overlap_stats_csv.write_header);
  writeCsv(instruction_timeline_csv.stream, instruction_timeline_data,
           instruction_timeline_csv.write_header);
  device_stats_csv.stream.close();
  comp_stats_csv.stream.close();
  comm_stats_csv.stream.close();
  overlap_stats_csv.stream.close();
  instruction_timeline_csv.stream.close();

  // Pipeline stats: write only if at least one Calcium arch produced rows
  // (PP > 1 was requested AND the cluster config is CalciumClusterConfig).
  // This keeps the existing CSV outputs byte-identical for non-PP runs.
  std::string pipeline_stats_csv_path;
  if (!pipeline_stats_data.empty()) {
    pipeline_stats_csv_path =
        tsl::io::JoinPath(output_path, "pipeline_stats.csv");
    CsvOut pipeline_stats_csv = createCsv(pipeline_stats_csv_path, true);
    if (pipeline_stats_csv.stream.is_open()) {
      // Prepend header.
      std::vector<std::vector<std::string>> with_header;
      with_header.reserve(pipeline_stats_data.size() + 1);
      with_header.push_back(pipeline_stats_header);
      for (auto& row : pipeline_stats_data) with_header.push_back(row);
      writeCsv(pipeline_stats_csv.stream, with_header,
               pipeline_stats_csv.write_header);
      pipeline_stats_csv.stream.close();
    } else {
      llvm::errs() << "Warning: failed to open pipeline_stats.csv for "
                      "writing; pipeline metrics dropped.\n";
      pipeline_stats_csv_path.clear();
    }
  }

  llvm::outs() << "CSV files saved to: "
               << tsl::io::JoinPath(output_path, "device_stats.csv") << ", "
               << tsl::io::JoinPath(output_path, "comp_stats.csv") << ", "
               << tsl::io::JoinPath(output_path, "comm_stats.csv") << ", "
               << tsl::io::JoinPath(output_path, "overlap_stats.csv") << ", "
               << tsl::io::JoinPath(output_path, "instruction_timeline.csv")
               << ", "
               << tsl::io::JoinPath(output_path, "compute_op_counts.csv")
               << ", "
               << tsl::io::JoinPath(output_path, "compute_memory_bound_stats.csv");
  if (!pipeline_stats_csv_path.empty()) {
    llvm::outs() << ", " << pipeline_stats_csv_path;
  }
  llvm::outs() << "\n";
  llvm::outs() << "Done\n";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
