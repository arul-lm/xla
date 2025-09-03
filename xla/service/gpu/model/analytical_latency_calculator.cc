// This file will be copied to a machine with XLA installed and run there.
#include "gpu_hlo_cost_analysis.h"
#include "gpu_performance_model.h"
#include "gpu_performance_model_base.h"
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
#include "xla/shape_util.h"
#include "xla/primitive_util.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include <cstdint>
#include <fstream>
#include <vector>
#include <set>
#include "absl/strings/str_split.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"
#include "absl/status/statusor.h"
#include "absl/status/status.h"
#include "tsl/platform/logging.h"



// Communication type enum for node-level analysis
enum class CommType {
  IntraNode,
  InterNode,
  Hierarchical
};



// Helper function to convert CommType enum to string
std::string CommTypeToString(CommType comm_type) {
  switch (comm_type) {
    case CommType::IntraNode:
      return "IntraNode";
    case CommType::InterNode:
      return "InterNode";
    case CommType::Hierarchical:
      return "Hierarchical";
    default:
      return "Unknown";
  }
}



// Structure to hold intra-node configuration
struct IntraNodeConfig {
  int size;           // Number of GPUs per node
  double bandwidth_gbps;  // Total bandwidth in GB/s shared between all GPUs in the node
  double efficiency_factor;  // Efficiency factor for collective operations (0.0 to 1.0)
  bool reassign_bandwidth;  // Whether to reassign bandwidth for this device type
};

// Device configuration structure
struct DeviceConfig {
  std::string name_pattern;
  int size;
  double bandwidth_gbps;
  double efficiency_factor;
  bool reassign_bandwidth;
  int links_per_chip;  // Number of ICI links per chip
};

// Global device configuration lookup table
static const std::vector<DeviceConfig> device_configs = {
    {"h100", 8, 900.0, 0.92, false, 0},
    {"h100_pcie", 8, 900.0, 0.92, false, 0},
    {"a100", 8, 600.0, 0.92, false, 0},
    {"v100", 8, 300.0, 0.88, false, 0},
    {"p100", 8, 160.0, 0.80, false, 0},
    {"a6000", 8, 64.0, 0.70, false, 0},
    {"mi200", 8, 800.0, 0.90, false, 0},
    {"mi300", 8, 1000.0, 0.93, false, 0},
    {"psg100", 512, 8000.0, 0.92, true, 0},
    {"tpuv4", 64, 90.0 * 6 / 2 * 64, 0.95, false, 6},
    {"tpuv5e", 64, 90.0 * 4 / 2 * 64, 0.95, false, 4},
    {"tpuv5p", 64, 180.0 * 6 / 2 * 64, 0.95, false, 6},
    {"tpuv6e", 64, 180.0 * 4 / 2 * 64, 0.95, false, 4},
    {"tpuv7e", 64, 200.0 * 4 / 2 * 64, 0.95, false, 4},
    {"tpuv7el200", 64, 2000.0 * 4 / 2 * 64, 0.95, false, 4},
    // {"tpuv7p", 64, 200.0 * 6 / 2 * 64, 0.95, false, 6}
};

// Structure to hold communication cost statistics
struct CommCostStats {
  CommType comm_type;        // Communication type (IntraNode/InterNode/Hierarchical)
  double comm_cost_us;       // Communication cost in microseconds
  double intranode_comm_vol_gb;   // Intra-node communication volume in GB
  double internode_comm_vol_gb;   // Inter-node communication volume in GB
  double intranode_comm_bw_gbps;  // Intra-node communication bandwidth in GB/s
  double internode_comm_bw_gbps;  // Inter-node communication bandwidth in GB/s
  double total_comm_vol_gb;       // Total communication volume in GB (sum of intranode and internode)
};

// Forward declarations
std::vector<int64_t> GetDeviceIdsFromOneReplicaGroup(const xla::HloInstruction* instr);
CommType DetermineCommType(const std::vector<int64_t>& device_ids, uint64_t replica_group_size, uint64_t intranode_size, const xla::HloInstruction* instr);
uint64_t GetNumReplicaGroups(const xla::HloInstruction* instr);
uint64_t GetReplicaGroupSize(const xla::HloInstruction* instr);
std::string GetDataTypeFromInstruction(const xla::HloInstruction* instr);
bool IsTpuDevice(const stream_executor::DeviceDescription& device_info);



// Helper function to validate IntraNodeConfig values
void ValidateIntraNodeConfig(const IntraNodeConfig& config) {
  CHECK_GE(config.size, 1) << "intranode_config.size must be at least 1, got: " << config.size;
  CHECK_GE(config.bandwidth_gbps, 1.0) << "intranode_config.bandwidth_gbps must be at least 1.0, got: " << config.bandwidth_gbps;
  CHECK_GT(config.efficiency_factor, 0.0) << "intranode_config.efficiency_factor must be greater than 0.0, got: " << config.efficiency_factor;
  CHECK_LE(config.efficiency_factor, 1.0) << "intranode_config.efficiency_factor must be less than or equal to 1.0, got: " << config.efficiency_factor;
}

// Helper function to determine intra-node size and bandwidth based on device type and compute capability
IntraNodeConfig GetIntraNodeConfigFromDeviceInfo(const stream_executor::DeviceDescription& gpu_device_info,
                                                 const std::string& fallback_device_type = "") {
  // Extract device name and compute capability
  std::string device_name = gpu_device_info.name();

  // Handle undefined device names
  if (device_name == "<undefined>" || device_name.empty()) {
    device_name = fallback_device_type;
  }

  // Search for exact device configuration match (case-insensitive)
  for (const auto& config : device_configs) {
    if (absl::AsciiStrToLower(device_name) == absl::AsciiStrToLower(config.name_pattern)) {
      IntraNodeConfig result = {config.size, config.bandwidth_gbps, config.efficiency_factor, config.reassign_bandwidth};
      ValidateIntraNodeConfig(result);
      return result;
    }
  }

  // Debug: Print available device patterns for troubleshooting
  llvm::outs() << "DEBUG: Device name '" << device_name << "' not found in device_configs.\n";
  llvm::outs() << "DEBUG: Available device patterns (exact match required):\n";
  for (const auto& config : device_configs) {
    llvm::outs() << "  - '" << config.name_pattern << "'\n";
  }
  llvm::outs().flush();

  // Fail with error if no exact device configuration match is found
  CHECK(false) << "No exact device configuration match found for device: '" << device_name
               << "'. Please add support for this device type in GetIntraNodeConfigFromDeviceInfo function.";
}

// Function to calculate communication cost based on device information and communication volume
CommCostStats CalculateCommCost(double per_device_comm_volume,
                               const stream_executor::DeviceDescription& gpu_device_info,
                               const xla::HloInstruction* instr, uint64_t replica_group_size,
                               uint64_t num_replica_groups,
                               const std::string& fallback_device_type = "",
                               double non_overlap_factor = 1.0) {
  // Get device-specific configuration (already validated in GetIntraNodeConfigFromDeviceInfo)
  IntraNodeConfig intranode_config = GetIntraNodeConfigFromDeviceInfo(gpu_device_info, fallback_device_type);

  // Validate per_device_comm_volume
  CHECK_GE(per_device_comm_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << per_device_comm_volume;

  // Validate replica_group_size and num_replica_groups
  CHECK_GE(replica_group_size, 1) << "replica_group_size must be at least 1, got: " << replica_group_size;
  CHECK_GE(num_replica_groups, 1) << "num_replica_groups must be at least 1, got: " << num_replica_groups;

  // Get device IDs to determine communication type
  auto device_ids = GetDeviceIdsFromOneReplicaGroup(instr);
  CommType comm_type = DetermineCommType(device_ids, replica_group_size, intranode_config.size, instr);

  // Initialize all fields
  double intranode_comm_vol_gb = 0.0;
  double internode_comm_vol_gb = 0.0;
  double intranode_comm_bw_gbps = (intranode_config.bandwidth_gbps * intranode_config.efficiency_factor) / intranode_config.size;
  intranode_comm_bw_gbps = intranode_comm_bw_gbps / 2.0; // Use only send bandwidth
  double internode_comm_bw_gbps = 10.0;
  double per_device_comm_vol_gb = per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);
  double comm_cost_us = 0.0;
  // Calculate per_device_comm_bw_gbps based on communication type
  if (comm_type == CommType::IntraNode) {
    if (intranode_config.reassign_bandwidth) {
      intranode_comm_bw_gbps = (intranode_config.bandwidth_gbps * intranode_config.efficiency_factor) / (num_replica_groups * replica_group_size);
    }
    intranode_comm_vol_gb = per_device_comm_vol_gb;
    comm_cost_us += intranode_comm_vol_gb / intranode_comm_bw_gbps * 1e6;

  } else if (comm_type == CommType::Hierarchical) {
    // Hierarchical happens because within a replica group, there are devices that go beyond the intranode_config.size
    // assert that replica_group_size is divisible by intranode_config.size
    CHECK(replica_group_size % intranode_config.size == 0) << "Replica group size is not divisible by intranode config size";
    auto num_intranode_groups = replica_group_size / intranode_config.size;

    // Multiple intra-node groups - hierarchical communication
    internode_comm_vol_gb = (per_device_comm_vol_gb / (intranode_config.size * num_intranode_groups)) * (num_intranode_groups - 1);
    intranode_comm_vol_gb = internode_comm_vol_gb * (intranode_config.size - 1);
    comm_cost_us += intranode_comm_vol_gb / intranode_comm_bw_gbps * 1e6;
    comm_cost_us += internode_comm_vol_gb / internode_comm_bw_gbps * 1e6;
  } else {
    // InterNode communication
    internode_comm_vol_gb = per_device_comm_vol_gb;
    comm_cost_us += internode_comm_vol_gb / internode_comm_bw_gbps * 1e6;
  }

  // Calculate total communication volume
  double total_comm_vol_gb = intranode_comm_vol_gb + internode_comm_vol_gb;

  // Apply non_overlap factor to communication cost
  comm_cost_us *= non_overlap_factor;

  // Debug logging to catch any remaining NaN values
  if (std::isnan(comm_cost_us) || std::isnan(intranode_comm_vol_gb) || std::isnan(internode_comm_vol_gb) ||
      std::isnan(intranode_comm_bw_gbps) || std::isnan(internode_comm_bw_gbps) || std::isnan(total_comm_vol_gb)) {
    llvm::outs() << "WARNING: NaN detected in CalculateCommCost for instruction: " << instr->ToString() << "\n";
    llvm::outs() << "  per_device_comm_volume: " << per_device_comm_volume << "\n";
    llvm::outs() << "  replica_group_size: " << replica_group_size << "\n";
    llvm::outs() << "  num_replica_groups: " << num_replica_groups << "\n";
    llvm::outs() << "  intranode_config.size: " << intranode_config.size << "\n";
    llvm::outs() << "  intranode_config.bandwidth_gbps: " << intranode_config.bandwidth_gbps << "\n";
    llvm::outs() << "  intranode_config.efficiency_factor: " << intranode_config.efficiency_factor << "\n";
    llvm::outs() << "  comm_type: " << CommTypeToString(comm_type) << "\n";
    llvm::outs() << "  comm_cost_us: " << comm_cost_us << "\n";
    llvm::outs() << "  intranode_comm_vol_gb: " << intranode_comm_vol_gb << "\n";
    llvm::outs() << "  internode_comm_vol_gb: " << internode_comm_vol_gb << "\n";
    llvm::outs() << "  intranode_comm_bw_gbps: " << intranode_comm_bw_gbps << "\n";
    llvm::outs() << "  internode_comm_bw_gbps: " << internode_comm_bw_gbps << "\n";
    llvm::outs() << "  non_overlap_factor: " << non_overlap_factor << "\n";
    llvm::outs().flush();
  }

  return {
    comm_type,
    comm_cost_us,
    intranode_comm_vol_gb,
    internode_comm_vol_gb,
    intranode_comm_bw_gbps,
    internode_comm_bw_gbps,
    total_comm_vol_gb
  };
}

// Helper function to convert device ID to 3D coordinates in torus
std::tuple<int, int, int> DeviceIdTo3DCoords(int64_t device_id, const std::vector<int>& mesh_shape) {
  CHECK_EQ(mesh_shape.size(), 3) << "Mesh shape must be 3D for TPU torus";
  int x_size = mesh_shape[0];
  int y_size = mesh_shape[1];
  int z_size = mesh_shape[2];

  int z = device_id % z_size;
  int y = (device_id / z_size) % y_size;
  int x = device_id / (y_size * z_size);

  return {x, y, z};
}

// Helper function to calculate Manhattan distance in 3D torus
int Calculate3DTorusDistance(int x1, int y1, int z1, int x2, int y2, int z2,
                            const std::vector<int>& mesh_shape) {
  int x_size = mesh_shape[0];
  int y_size = mesh_shape[1];
  int z_size = mesh_shape[2];

  // Calculate distance in each dimension considering torus wraparound
  int dx = std::min(std::abs(x2 - x1), std::min(std::abs(x2 - x1 + x_size), std::abs(x2 - x1 - x_size)));
  int dy = std::min(std::abs(y2 - y1), std::min(std::abs(y2 - y1 + y_size), std::abs(y2 - y1 - y_size)));
  int dz = std::min(std::abs(z2 - z1), std::min(std::abs(z2 - z1 + z_size), std::abs(z2 - z1 - z_size)));

  return dx + dy + dz;
}

// Helper function to calculate maximum distance between devices in replica group
int CalculateMaxDistanceInReplicaGroup(const std::vector<int64_t>& device_ids,
                                      const std::vector<int>& mesh_shape) {
  int max_distance = 0;

  for (size_t i = 0; i < device_ids.size(); ++i) {
    for (size_t j = i + 1; j < device_ids.size(); ++j) {
      auto [x1, y1, z1] = DeviceIdTo3DCoords(device_ids[i], mesh_shape);
      auto [x2, y2, z2] = DeviceIdTo3DCoords(device_ids[j], mesh_shape);

      int distance = Calculate3DTorusDistance(x1, y1, z1, x2, y2, z2, mesh_shape);
      max_distance = std::max(max_distance, distance);
    }
  }

  return max_distance;
}

// Helper function to estimate number of hops for all-reduce in 3D torus
int EstimateAllReduceHops(int replica_group_size, int max_distance, const std::vector<int>& mesh_shape) {
  // For all-reduce in 3D torus, we need to consider:
  // 1. Reduce-scatter phase: log2(replica_group_size) steps
  // 2. All-gather phase: log2(replica_group_size) steps
  // 3. Each step involves communication across max_distance

  int reduce_scatter_hops = static_cast<int>(std::log2(replica_group_size)) * max_distance;
  int all_gather_hops = static_cast<int>(std::log2(replica_group_size)) * max_distance;

  return reduce_scatter_hops + all_gather_hops;
}

// Function to calculate TPU communication cost based on 3D torus topology
CommCostStats CalculateTpuCommCost(double per_device_comm_volume,
                                   const stream_executor::DeviceDescription& tpu_device_info,
                                   const xla::HloInstruction* instr, uint64_t replica_group_size,
                                   uint64_t num_replica_groups,
                                   const std::vector<int64_t>& device_ids,
                                   const std::vector<int>& mesh_shape,
                                   const std::string& fallback_device_type = "",
                                   double non_overlap_factor = 1.0) {
  // Validate inputs
  CHECK_GE(per_device_comm_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << per_device_comm_volume;
  CHECK_GE(replica_group_size, 1) << "replica_group_size must be at least 1, got: " << replica_group_size;
  CHECK_GE(num_replica_groups, 1) << "num_replica_groups must be at least 1, got: " << num_replica_groups;
  CHECK_EQ(mesh_shape.size(), 3) << "Mesh shape must be 3D for TPU torus";
  CHECK_EQ(device_ids.size(), replica_group_size) << "Device IDs size must match replica group size";

  // Get device-specific ICI bandwidth from device configuration
  IntraNodeConfig intranode_config = GetIntraNodeConfigFromDeviceInfo(tpu_device_info, fallback_device_type);

  // Get the device configuration to access links_per_chip
  std::string device_name = tpu_device_info.name();
  int links_per_chip = 6;  // Default for TPU v4

  // Find the device configuration to get links_per_chip
  for (const auto& config : device_configs) {
    if (absl::StrContains(absl::AsciiStrToLower(device_name), absl::AsciiStrToLower(config.name_pattern))) {
      links_per_chip = config.links_per_chip;
      break;
    }
  }

  // Calculate ICI bandwidth per hop (per link)
  // per-link bandwidth = total_bandwidth / (links_per_chip * num_chips)
  double ici_bandwidth_per_link_gbps = intranode_config.bandwidth_gbps / (links_per_chip * 64.0);


  // Calculate maximum distance between two neighbors in the replica group
  int max_distance = CalculateMaxDistanceInReplicaGroup(device_ids, mesh_shape);

  // Calculate number of hops to reach the neighbor at max_distance
  int number_of_hops = max_distance;  // Direct hop count to reach the neighbor

  // Calculate communication volume per device in GB
  double per_device_comm_vol_gb = per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  // Calculate communication cost using the formula:
  // cost = (per_device_comm_gb / send_link_bw) * number_of_hops
  double comm_cost_us = (per_device_comm_vol_gb / ici_bandwidth_per_link_gbps) * number_of_hops * 1e6;

  // Apply non_overlap factor
  comm_cost_us *= non_overlap_factor;

  // For TPU, all communication is intra-pod (within the 3D torus)
  double intranode_comm_vol_gb = per_device_comm_vol_gb;
  double internode_comm_vol_gb = 0.0;
  double total_comm_vol_gb = intranode_comm_vol_gb;

  // Determine communication type (all TPU communication is intra-pod for now)
  CommType comm_type = CommType::IntraNode;

  // Debug logging
  if (std::isnan(comm_cost_us) || std::isnan(intranode_comm_vol_gb) || std::isnan(total_comm_vol_gb)) {
    llvm::outs() << "WARNING: NaN detected in CalculateTpuCommCost for instruction: " << instr->ToString() << "\n";
    llvm::outs() << "  per_device_comm_volume: " << per_device_comm_volume << "\n";
    llvm::outs() << "  replica_group_size: " << replica_group_size << "\n";
    llvm::outs() << "  max_distance: " << max_distance << "\n";
    llvm::outs() << "  links_per_chip: " << links_per_chip << "\n";
    llvm::outs() << "  number_of_hops: " << number_of_hops << "\n";
    llvm::outs() << "  ici_bandwidth_per_link_gbps: " << ici_bandwidth_per_link_gbps << "\n";
    llvm::outs() << "  comm_cost_us: " << comm_cost_us << "\n";
    llvm::outs().flush();
  }

  return {
    comm_type,
    comm_cost_us,
    intranode_comm_vol_gb,
    internode_comm_vol_gb,
    ici_bandwidth_per_link_gbps,  // intranode_comm_bw_gbps
    0.0,                          // internode_comm_bw_gbps (not used for TPU)
    total_comm_vol_gb
  };
}

struct CliOpts {
  std::string hlo_module_file;
  std::string comm_non_overlap = "1.0";
  std::vector<std::string> hardware_architectures;
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

// Helper function to validate hardware architectures and check if config files exist
absl::Status ValidateHardwareArchitectures(const std::vector<std::string>& hardware_architectures) {
  if (hardware_architectures.empty()) {
    return absl::InvalidArgumentError("No hardware architectures specified. Use --hardware-architectures to specify at least one architecture.");
  }

  std::vector<std::string> invalid_architectures;

  for (const std::string& arch : hardware_architectures) {
    if (arch.empty()) {
      invalid_architectures.push_back("(empty string)");
      continue;
    }

    // Construct the expected config file path
    auto config_path = tsl::io::JoinPath("/xla", "xla", "tools", "hlo_opt", "gpu_specs",
                                        absl::StrCat(arch, ".txtpb"));

    // Check if the config file exists
    absl::Status file_exists_status = tsl::Env::Default()->FileExists(config_path);
    if (!file_exists_status.ok()) {
      invalid_architectures.push_back(arch);
    }
  }

  if (!invalid_architectures.empty()) {
    std::string error_msg = "Invalid hardware architectures - config files not found:\n";
    for (const std::string& invalid_arch : invalid_architectures) {
      auto expected_path = tsl::io::JoinPath("/xla", "xla", "tools", "hlo_opt", "gpu_specs",
                                           absl::StrCat(invalid_arch, ".txtpb"));
      error_msg += absl::StrCat("  - ", invalid_arch, " (expected: ", expected_path, ")\n");
    }
    error_msg += "\nAvailable hardware architectures can be found in /xla/xla/tools/hlo_opt/gpu_specs/";
    return absl::InvalidArgumentError(error_msg);
  }

  return absl::OkStatus();
}





// Helper function to get device count from replica groups
absl::StatusOr<int64_t> GetCollectiveOpDeviceCount(const xla::HloInstruction* inst) {
  switch (inst->opcode()) {
    case xla::HloOpcode::kAllReduce:
    case xla::HloOpcode::kAllGather:
    case xla::HloOpcode::kAllToAll:
    case xla::HloOpcode::kReduceScatter: {
      const auto* collective_inst = xla::Cast<xla::HloCollectiveInstruction>(inst);
      const auto& replica_groups = collective_inst->replica_groups();
      // There are no replica groups. So, all devices are involved.
      if (replica_groups.empty()) {
        return inst->GetModule()->config().replica_count();
      }
      const size_t subgroup_size = replica_groups[0].replica_ids_size();
      for (size_t i = 1; i < replica_groups.size(); ++i) {
        CHECK_EQ(replica_groups[i].replica_ids_size(), subgroup_size)
            << "Expected symmetric replica groups for instruction: " << inst->ToString();
      }
      return subgroup_size;
    }
    case xla::HloOpcode::kCollectivePermute: {
      const auto* collective_permute = xla::Cast<xla::HloCollectivePermuteInstruction>(inst);
      return collective_permute->source_target_pairs().size();
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("GetCollectiveOpDeviceCount not implemented for opcode: ",
          xla::HloOpcodeString(inst->opcode())));
  }
}

// Helper function to get device IDs from one replica group after symmetry check
std::vector<int64_t> GetDeviceIdsFromOneReplicaGroup(const xla::HloInstruction* instr) {
  std::vector<int64_t> device_ids;

  // Handle CollectivePermute separately since it doesn't inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto* collective_permute = xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto& pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      device_ids.assign(participants.begin(), participants.end());
    }
    return device_ids;
  }

  // Handle other collective operations that inherit from HloCollectiveInstruction
  // Only try to cast for operations that actually inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto* collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto& replica_groups = collective_inst->replica_groups();

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
      const auto& first_group = replica_groups[0];
      for (size_t i = 0; i < first_group.replica_ids_size(); ++i) {
        device_ids.push_back(first_group.replica_ids(i));
      }
    }
  }

  return device_ids;
}

// Helper function to get number of replica groups from instruction
uint64_t GetNumReplicaGroups(const xla::HloInstruction* instr) {
  // Handle CollectivePermute separately since it doesn't inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto* collective_permute = xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto& pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      // For CollectivePermute, we consider it as one replica group
      return 1;
    }
    return 0;
  }

  // Handle other collective operations that inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto* collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto& replica_groups = collective_inst->replica_groups();

      if (replica_groups.empty()) {
        return 0;
      }

      // Return the number of replica groups
      uint64_t num_groups = replica_groups.size();
      CHECK_GE(num_groups, 1) << "num_replica_groups must be at least 1, got: " << num_groups;
      return num_groups;
    }
  }

  return 0;
}

// Helper function to get replica group size from instruction
uint64_t GetReplicaGroupSize(const xla::HloInstruction* instr) {
  // Handle CollectivePermute separately since it doesn't inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    const auto* collective_permute = xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
    if (collective_permute) {
      std::set<int64_t> participants;
      for (const auto& pair : collective_permute->source_target_pairs()) {
        participants.insert(pair.first);
        participants.insert(pair.second);
      }
      uint64_t size = participants.size();
      CHECK_GE(size, 1) << "replica_group_size must be at least 1, got: " << size;
      return size;
    }
    return 0;
  }

  // Handle other collective operations that inherit from HloCollectiveInstruction
  if (instr->opcode() == xla::HloOpcode::kAllReduce ||
      instr->opcode() == xla::HloOpcode::kAllGather ||
      instr->opcode() == xla::HloOpcode::kReduceScatter ||
      instr->opcode() == xla::HloOpcode::kAllToAll) {

    auto* collective_inst = xla::Cast<xla::HloCollectiveInstruction>(instr);
    if (collective_inst) {
      const auto& replica_groups = collective_inst->replica_groups();

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

      CHECK_GE(first_group_size, 1) << "replica_group_size must be at least 1, got: " << first_group_size;
      return first_group_size;
    }
  }

  return 0;
}

// Helper function to determine communication type based on device IDs
CommType DetermineCommType(const std::vector<int64_t>& device_ids, uint64_t replica_group_size, uint64_t intranode_size, const xla::HloInstruction* instr) {
  // FIXME: Intentional oversimplification - CollectivePermute operations are always classified as InterNode
  // This is a temporary simplification that should be refined to properly analyze the source-target pairs
  // and determine if the permutation pattern is truly inter-node or could be intra-node
  if (instr->opcode() == xla::HloOpcode::kCollectivePermute) {
    return CommType::InterNode;
  }

  if (device_ids.empty()) {
    // Handle empty device_ids case - needs to be fixed later
    // For now, check total number of devices involved in the collective operation
    auto device_count_result = GetCollectiveOpDeviceCount(instr);
    int total_devices = device_count_result.ok() ? static_cast<int>(device_count_result.value()) : 1;
    CHECK(false) << "Empty device_ids vector encountered. Total devices: " << total_devices
                 << ". This case needs to be fixed later for instruction: " << instr->ToString();
  }

  if (device_ids.size() < 2) {
    // Single device or no devices - consider as intra-node
    return CommType::IntraNode;
  }

  // Sort device IDs to ensure proper distance calculation
  std::vector<int64_t> sorted_device_ids = device_ids;
  std::sort(sorted_device_ids.begin(), sorted_device_ids.end());

  // Check distance between first two devices
  int64_t first_distance = sorted_device_ids[1] - sorted_device_ids[0];

  // Check if all devices are of the same distance from their neighbors
  for (size_t i = 2; i < sorted_device_ids.size(); ++i) {
    int64_t distance = sorted_device_ids[i] - sorted_device_ids[i-1];
    if (distance != first_distance) {
      // Uneven spacing - this is unexpected for now, fail with error
      CHECK(false) << "Uneven device spacing detected. First distance: " << first_distance
                   << ", distance at index " << i << ": " << distance
                   << ". Device IDs: [";
      for (size_t j = 0; j < sorted_device_ids.size(); ++j) {
        if (j > 0) llvm::outs() << ", ";
        llvm::outs() << sorted_device_ids[j];
      }
      llvm::outs() << "]. This case needs to be handled properly.";
    }
  }

  // All devices have the same spacing, check if it's less than intranode_size
  if (first_distance < intranode_size) {
    if (replica_group_size <= intranode_size) {
      return CommType::IntraNode;
    } else {
      return CommType::Hierarchical;
    }
  } else {
    return CommType::InterNode;
  }

  return CommType::IntraNode;
}

// Helper function to get device count (wrapper for backward compatibility)
uint64_t GetDeviceCount(const xla::HloInstruction* instr) {
  auto result = GetCollectiveOpDeviceCount(instr);
  if (result.ok()) {
    return static_cast<uint64_t>(result.value());
  }
  return 1; // Default fallback
}

// Helper function to detect if device is TPU
bool IsTpuDevice(const stream_executor::DeviceDescription& device_info) {
  std::string name = device_info.name();
  std::string lower_name = absl::AsciiStrToLower(name);
  return absl::StrContains(lower_name, "tpu");
}

// Helper function to get data type from instruction result shape
std::string GetDataTypeFromInstruction(const xla::HloInstruction* instr) {
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
    default:
      return "unknown";
  }
}

// Helper function to add computation statistics to CSV data
void AddCompStatsToCSV(std::vector<std::vector<std::string>>& comp_stats_data,
                      int comp_id, int inst_count, const std::string& deduplicated_name,
                      double cost, const xla::gpu::EstimateRunTimeData& runtime_data,
                      const std::string& device_name, const xla::HloInstruction* instr) {
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
    throughput = (runtime_data.flops / exec_time_seconds) / 1e12; // Convert to TFLOPs/s
  }

  comp_stats_data.push_back({
    std::to_string(comp_id), std::to_string(inst_count), std::string(deduplicated_name),
    group_name, std::to_string(cost),
    std::to_string(runtime_data.flops / 1e12), // Convert to TFLOPs
    std::to_string(runtime_data.bytes_read / 1e9), // Convert to GB
    std::to_string(runtime_data.bytes_written / 1e9), // Convert to GB
    absl::FormatDuration(runtime_data.compute_time),
    absl::FormatDuration(runtime_data.read_time),
    absl::FormatDuration(runtime_data.write_time),
    std::to_string(throughput), device_name, GetDataTypeFromInstruction(instr)
  });
}







// Communication volume calculation with topology support
struct CommunicationVolume {
  double per_device_volume;
  double total_volume;
  std::string pattern;
  std::string formula;
};

// Helper function to calculate ring-based communication volume
CommunicationVolume CalculateRingVolume(double per_step_data, uint64_t replica_group_size,
                                       const std::string& operation_name,
                                       const std::string& formula, double multiplier = 1.0) {
  double total_steps = replica_group_size - 1;
  double per_device_volume = per_step_data * total_steps;
  double total_volume = per_device_volume * replica_group_size;

  return {
    per_device_volume * multiplier,
    total_volume * multiplier,
    "Ring-based " + operation_name,
    formula
  };
}

CommunicationVolume CalculateCommunicationVolume(const xla::HloInstruction* instr,
                                                 uint64_t replica_group_size) {

  double operand_size = static_cast<double>(xla::ShapeUtil::ByteSizeOf(instr->operand(0)->shape()));
  double result_size = static_cast<double>(xla::ShapeUtil::ByteSizeOf(instr->shape()));

  switch (instr->opcode()) {
    case xla::HloOpcode::kAllReduce: {
        CHECK_EQ(operand_size, result_size)
            << "AllReduce operand and result sizes must be equal. "
            << "Operand size: " << operand_size << ", Result size: " << result_size
            << " for instruction: " << instr->ToString();

        double per_step_data = operand_size / replica_group_size;
        std::string formula = "operand_size/" + std::to_string(replica_group_size) + " * (" + std::to_string(replica_group_size) + "-1)";
        CommunicationVolume result = CalculateRingVolume(per_step_data, replica_group_size, "AllReduce", formula, 2.0);
        CHECK_GT(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
        return result;
    }
    case xla::HloOpcode::kAllGather: {
        CHECK_GE(result_size, operand_size)
            << "AllGather result size must be greater than or equal to operand size. "
            << "Operand size: " << operand_size << ", Result size: " << result_size
            << " for instruction: " << instr->ToString();

        std::string formula = "operand_size * (" + std::to_string(replica_group_size) + "-1)";
        CommunicationVolume result = CalculateRingVolume(operand_size, replica_group_size, "AllGather", formula);
        CHECK_GT(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
        return result;
    }
    case xla::HloOpcode::kReduceScatter: {
        CHECK_LE(result_size, operand_size)
            << "ReduceScatter result size must be less than or equal to operand size. "
            << "Operand size: " << operand_size << ", Result size: " << result_size
            << " for instruction: " << instr->ToString();

        if (operand_size == result_size) {
            CHECK_EQ(replica_group_size, 1) << "In Reduce scatter, when operand_size == result_size, replica_group_size should be 1";
        }

        std::string formula = "result_size * (" + std::to_string(replica_group_size) + "-1)";
        CommunicationVolume result = CalculateRingVolume(result_size, replica_group_size, "ReduceScatter", formula);
        if (result.per_device_volume == 0.0) {
            llvm::outs() << "WARN. Local reduce-scatter operation detected. "
                         << "Instruction: " << instr->ToString() << ", "
                         << "replica_group_size: " << replica_group_size << ", "
                         << "operand_size: " << operand_size << "\n";
            llvm::outs().flush();
        }
        CHECK_GE(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
        return result;
    }
    case xla::HloOpcode::kCollectivePermute: {
      CHECK_EQ(operand_size, result_size)
          << "CollectivePermute operand and result sizes must be equal. "
          << "Operand size: " << operand_size << ", Result size: " << result_size
          << " for instruction: " << instr->ToString();

      CommunicationVolume result = {
        operand_size,
        operand_size * 2,
        "Point-to-point permutation",
        "operand_size * 2"
      };
      CHECK_GT(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
      return result;
    }
    default:
      return {0.0, 0.0, "Unknown", "Unknown"};
  }
}

int main(int argc, char* argv[]) {
  llvm::errs().tie(&llvm::outs());
  CliOpts opts;
  std::string hardware_architectures_str;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo-module-file", &opts.hlo_module_file, "Filename of HloModule"),
      tsl::Flag("comm-non-overlap", &opts.comm_non_overlap, "Communication non-overlap factor (0.0 to 1.0)"),
      tsl::Flag("hardware-architectures", &hardware_architectures_str, "Comma-separated list of hardware architectures (e.g., h100_pcie,tpuv5p,tpuv6e)")};
  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage_string = tsl::Flags::Usage(argv[0], flag_list);
  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    return 1;
  }

  // Parse hardware architectures from comma-separated string
  if (!hardware_architectures_str.empty()) {
    std::vector<std::string> arch_list = absl::StrSplit(hardware_architectures_str, ',');
    for (std::string& arch : arch_list) {
      // Trim whitespace
      arch = absl::StripAsciiWhitespace(arch);
      if (!arch.empty()) {
        opts.hardware_architectures.push_back(arch);
      }
    }
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  CHECK(opts.hlo_module_file.empty() == false)
      << "Path to HLO module file required";

  // Validate comm_non_overlap factor
  double comm_non_overlap_value = std::stod(opts.comm_non_overlap);
  CHECK_GE(comm_non_overlap_value, 0.0) << "comm_non_overlap must be at least 0.0, got: " << comm_non_overlap_value;
  CHECK_LE(comm_non_overlap_value, 1.0) << "comm_non_overlap must be at most 1.0, got: " << comm_non_overlap_value;

  // Validate hardware architectures
  absl::Status validation_status = ValidateHardwareArchitectures(opts.hardware_architectures);
  if (!validation_status.ok()) {
    llvm::outs() << "Error: " << validation_status.message() << "\n";
    llvm::outs().flush();
    return 1;
  }

  llvm::outs() << "Using comm_non_overlap factor: " << comm_non_overlap_value << "\n";
  llvm::outs() << "Using hardware architectures: ";
  for (size_t i = 0; i < opts.hardware_architectures.size(); ++i) {
    if (i > 0) llvm::outs() << ", ";
    llvm::outs() << opts.hardware_architectures[i];
  }
  llvm::outs() << "\n";
  llvm::outs().flush();

  std::string prefix = "/xla";
  auto stats_path = tsl::io::JoinPath("/xla", "stats");
  std::ofstream device_stats_csv =
      createCsv(tsl::io::JoinPath(stats_path, "device_stats.csv"));
  auto comp_stats_csv = createCsv(tsl::io::JoinPath(stats_path, "comp_stats.csv"));
  auto comm_stats_csv = createCsv(tsl::io::JoinPath(stats_path, "comm_stats.csv"));
  if (!device_stats_csv.is_open() || !comp_stats_csv.is_open() || !comm_stats_csv.is_open()) {
    return -1;
  }

  std::vector<std::vector<std::string>> device_stats_data, comp_stats_data, comm_stats_data;
  std::vector<std::string> device_stats_header = {"device_name", "latency"};
  std::vector<std::string> comp_stats_header = {
      "comp_id",   "idx",       "inst",       "group",         "latency(µs)",
      "tflops",    "bytes_read_gb", "bytes_written_gb", "compute_time",
      "read_time", "write_time", "throughput_tflops_per_sec", "device_name", "datatype"};
  std::vector<std::string> comm_stats_header = {
      "comp_id",   "instruction_name", "opcode", "device_name", "comm_type",
      "comm_cost_us", "replica_group_size", "num_replica_groups",
      "intranode_comm_vol_gb", "internode_comm_vol_gb", "intranode_comm_bw_gbps", "internode_comm_bw_gbps",
      "total_comm_vol_gb", "datatype"};
  device_stats_data.push_back(device_stats_header);
  comp_stats_data.push_back(comp_stats_header);
  comm_stats_data.push_back(comm_stats_header);
  std::string format = "hlo";
  std::unique_ptr<xla::HloModule> hlo_module =
      *xla::LoadModuleFromFile(opts.hlo_module_file, format, {});

  absl::flat_hash_map<std::string, std::pair<stream_executor::GpuDeviceInfoProto, std::string>>
      gpu_specs;
  for (const std::string& file_name : opts.hardware_architectures) {
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
      llvm::outs() << "Device spec file read failed for: " << path << "\n";
      return -1;
    }
    llvm::outs().flush();
    gpu_specs[proto.device_description_str()] = std::make_pair(proto.gpu_device_info(), file_name);
  }

  auto const SEPARATOR = ",";
  int64_t pointer_size = 8;
  for (const auto& pair : gpu_specs) {
    auto device_name = pair.first;
    stream_executor::GpuDeviceInfoProto gpu_device_info_pb = pair.second.first;
    std::string spec_file_name = pair.second.second;
    absl::StatusOr<stream_executor::DeviceDescription> gpu_device_info_result =
        stream_executor::DeviceDescription::FromProto(gpu_device_info_pb);

    if (!gpu_device_info_result.ok()) {
      llvm::outs() << "Failed to create DeviceDescription: " << device_name << "\n";
      llvm::outs().flush();
      return -1;
    }

    stream_executor::DeviceDescription gpu_device_info = gpu_device_info_result.value();
    // uint64_t memory_limit = xla::gpu::GetSchedulerMemoryLimit(
    //     *hlo_module, gpu_device_info, pointer_size);
    // std::cout << "Mem validation passed. Memory limit(in GB):"
    //           << memory_limit / 1e9 << "\n";
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
    // auto ale_latency_estimator =
    //     std::make_unique<xla::gpu::AnalyticalLatencyEstimator>(
    //         scheduler_config, std::move(gle_latency_estimator),
    //         gpu_device_info, xla::HloCostAnalysis::DefaultShapeSize,
    //         hlo_module->entry_computation());
    xla::gpu::GpuPerformanceModelOwning gpu_performance_model_(gpu_device_info);

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
    TF_CHECK_OK(
        hlo_module->entry_computation()->Accept(ale_cost_analysis.get()));
    TF_CHECK_OK(
        hlo_module->entry_computation()->Accept(coll_cost_analysis.get()));

    auto gpu_latency_estimator =
        std::make_unique<xla::gpu::GpuLatencyEstimator>(pointer_size);

    auto inst_count = 0;
    auto comp_count = 0;
    auto comp_cost = 0.0;
    auto entry_comp_cost = 0.0;  // Track cost of entry computation specifically
    auto entry_compute_time = 0.0;  // Track compute time for entry computation
    auto entry_comm_time = 0.0;  // Track communication time for entry computation
    absl::flat_hash_map<const xla::HloComputation*, double> computation_map;
    absl::flat_hash_map<const xla::HloComputation*, int> computation_idx;
    for (xla::HloComputation* computation : hlo_module->computations()) {
      comp_cost = 0;
      auto is_entry_comp = computation->IsEntryComputation();
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
        xla::gpu::EstimateRunTimeData runtime_data{};
        switch (opcode) {
          case xla::HloOpcode::kWhile: {
            auto comps = {instr->while_body(), instr->while_condition()};
            auto while_cost = 0.0;
            for (auto comp : comps) {
              auto it = computation_map.find(comp);
              auto it2 = computation_idx.find(comp);
              llvm::outs() << "Adding cost of comp:" << it2->second << "\n";
              CHECK(it != computation_map.end());
              while_cost += it->second;
            }
            absl::StatusOr<xla::WhileLoopBackendConfig> while_config =
                instr->backend_config<xla::WhileLoopBackendConfig>();
            if (while_config.ok() && while_config->has_known_trip_count()) {
                cost = while_cost * while_config->known_trip_count().n();
            } else {
              CHECK(1 == 0);
            }

            // Track compute time for entry computation
            if (is_entry_comp) {
              entry_compute_time += cost;
            }

            // FIXME: Runtime_data will be outdated
            runtime_data =
                gpu_performance_model_.Get().EstimateRunTimeForInstruction(
                    instr, &*ale_cost_analysis);

            // Add computation statistics to CSV data for all computations
            if (cost > 0) {
              inst_count += 1;
              AddCompStatsToCSV(comp_stats_data, comp_count, inst_count, std::string(deduplicated_name),
                               cost, runtime_data, device_name, instr);
            }
            break;
          }
          // case xla::HloOpcode::kWhile: {
          //   runtime_data =
          //   gpu_performance_model_.EstimateRunTimeForInstruction(
          //       instr, &*ale_cost_analysis);
          //   cost = absl::ToDoubleMicroseconds(runtime_data.exec_time);
          //   // cost = ale_latency_estimator->NodeCost(instr);
          //   // llvm::outs() << "While loop cost:" << cost << "\n";
          //   // llvm::outs().flush();
          //   break;
          // }
          case xla::HloOpcode::kReduceScatter:
          case xla::HloOpcode::kCollectivePermute:
          case xla::HloOpcode::kAllGather:
          case xla::HloOpcode::kAllReduce: {

            // Calculate communication cost statistics
            uint64_t replica_group_size = GetReplicaGroupSize(instr);
            uint64_t num_replica_groups = GetNumReplicaGroups(instr);
            CommunicationVolume comm_volume = CalculateCommunicationVolume(instr, replica_group_size);

            // Dispatch to appropriate communication cost calculation based on device type
            CommCostStats comm_stats;
            if (IsTpuDevice(gpu_device_info)) {
              // Get device IDs and mesh shape for TPU communication cost calculation
              auto device_ids = GetDeviceIdsFromOneReplicaGroup(instr);
              // Default TPU v4 pod slice mesh shape: 4x4x4
              std::vector<int> mesh_shape = {4, 4, 4};
              comm_stats = CalculateTpuCommCost(comm_volume.per_device_volume, gpu_device_info, instr, replica_group_size, num_replica_groups, device_ids, mesh_shape, spec_file_name, comm_non_overlap_value);
            } else {
              comm_stats = CalculateCommCost(comm_volume.per_device_volume, gpu_device_info, instr, replica_group_size, num_replica_groups, spec_file_name, comm_non_overlap_value);
            }
            cost = comm_stats.comm_cost_us;

            // Track communication time for entry computation
            if (is_entry_comp) {
              entry_comm_time += cost;
            }

            // std::vector<std::string> comm_stats_header = {
            //     "comp_id",   "instruction_name", "opcode", "device_name", "comm_type",
            //     "comm_volume_gb", "comm_bandwidth_gbps", "comm_cost_us", "replica_group_size"};
            // Add communication statistics to CSV data
            std::vector<std::string> comm_row = {
                std::to_string(comp_count),
                std::string(deduplicated_name),
                std::string(xla::HloOpcodeString(opcode)),
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
                GetDataTypeFromInstruction(instr)
            };
            comm_stats_data.push_back(comm_row);

            // Add computation statistics to CSV data for all computations
            if (cost > 0) {
              inst_count += 1;
            //   AddCompStatsToCSV(comp_stats_data, comp_count, inst_count, std::string(deduplicated_name),
            //                    cost, runtime_data, device_name);
            }
            break;
          }
          case xla::HloOpcode::kDot: {
              runtime_data = gpu_performance_model_.Get().EstimateRunTimeForInstruction(
                instr, &*ale_cost_analysis);

            cost = absl::ToDoubleMicroseconds(runtime_data.exec_time);

            // Track compute time for entry computation
            if (is_entry_comp) {
              entry_compute_time += cost;
            }




            // Add computation statistics to CSV data for all computations
            if (cost > 0) {
              inst_count += 1;
              AddCompStatsToCSV(comp_stats_data, comp_count, inst_count, std::string(deduplicated_name),
                               cost, runtime_data, device_name, instr);
            }

            // cost = ale_latency_estimator->NodeCost(instr);
            break;
          }
          // case xla::HloOpcode::kCopy:
          // case xla::HloOpcode::kCopyStart:
          // case xla::HloOpcode::kCopyDone:
          // case xla::HloOpcode::kDynamicSlice:
          // case xla::HloOpcode::kSlice:
          // case xla::HloOpcode::kDynamicUpdateSlice: {
          //   cost = ale_latency_estimator->NodeCost(instr);
          //   break;
          // }
          default: {
            // cost = ale_latency_estimator->NodeCost(instr);
            cost = 0.0;
            break;
          }
        };                      // Switch ends here
        comp_cost += cost;
      } // Inst for loop ends

      // Track entry computation cost separately
      if (is_entry_comp) {
        entry_comp_cost = comp_cost;
      }

      computation_map.insert({computation, comp_cost});
      computation_idx.insert({computation, comp_count});
      comp_count += 1;
    }  // Comp for loop ends

    // Print entry computation timing breakdown
    llvm::outs() << "=== Entry Computation Timing Breakdown ===\n";
    llvm::outs() << "Device: " << device_name << "\n";
    llvm::outs() << "Total Compute Time: " << entry_compute_time / 1e6 << " seconds\n";
    llvm::outs() << "Total Communication Time: " << entry_comm_time / 1e6 << " seconds\n";
    llvm::outs() << "Total Time: " << entry_comp_cost / 1e6 << " seconds\n";
    llvm::outs() << "==========================================\n";
    llvm::outs().flush();

    // Use entry computation cost for device_stats
    auto entry_comp_cost_in_secs = entry_comp_cost / 1e6;
    device_stats_data.push_back(
        {device_name, std::to_string(entry_comp_cost_in_secs)});
    llvm::outs().flush();
  }
  writeCsv(device_stats_csv, device_stats_data);
  writeCsv(comp_stats_csv, comp_stats_data);
  writeCsv(comm_stats_csv, comm_stats_data);
  device_stats_csv.close();
  comp_stats_csv.close();
  comm_stats_csv.close();
  llvm::outs() << "CSV files saved to: " << tsl::io::JoinPath(stats_path, "device_stats.csv")
               << ", " << tsl::io::JoinPath(stats_path, "comp_stats.csv")
               << ", " << tsl::io::JoinPath(stats_path, "comm_stats.csv") << "\n";
  llvm::outs() << "Done\n";
  return 0;
}
