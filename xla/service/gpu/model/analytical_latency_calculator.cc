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
#include <cmath>


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
  double intranode_bandwidth_gbps;  // Intra-node unidirectional bandwidth in GB/s
  double internode_bandwidth_gbps;  // Inter-node unidirectional bandwidth in GB/s
  double intranode_efficiency_factor;  // Efficiency factor for intra-node communication
  double internode_efficiency_factor;  // Efficiency factor for inter-node communication
  bool reassign_bandwidth;
  int links_per_chip;  // Number of ICI links per chip
  std::vector<int> intranode_mesh_shape;  // Physical mesh shape for intra-node (pod slice) torus topology
  std::vector<int> internode_mesh_shape;  // Physical mesh shape for inter-node (pod arrangement) torus topology
};

// Global device configuration lookup table
static const std::vector<DeviceConfig> device_configs = {
    {"h100", 900.0, 10.0, 0.92, 0.85, false, 0, {}, {}},
    {"h100_pcie", 900.0, 10.0, 0.92, 0.85, false, 0, {}, {}},
    {"a100", 600.0, 10.0, 0.92, 0.85, false, 0, {}, {}},
    {"v100", 300.0, 10.0, 0.88, 0.80, false, 0, {}, {}},
    {"p100", 160.0, 10.0, 0.80, 0.75, false, 0, {}, {}},
    {"a6000", 64.0, 10.0, 0.70, 0.65, false, 0, {}, {}},
    {"mi200", 800.0, 10.0, 0.90, 0.85, false, 0, {}, {}},
    {"mi300", 1000.0, 10.0, 0.93, 0.88, false, 0, {}, {}},
    {"psg100", 8000.0, 10.0, 0.92, 0.85, true, 0, {}, {}},
    {"tpuv4", 90.0 / 2, 90.0 / 2, 0.95, 0.85, false, 6, {4, 4, 4}, {4, 4, 4}},
    {"tpuv5e", 90.0 / 2, 90.0 / 2, 0.95, 0.85, false, 4, {4, 4, 4}, {4, 4, 4}},
    {"tpuv5p", 180.0 / 2, 180.0 / 2, 0.95, 0.85, false, 6, {4, 4, 4}, {4, 4, 4}},
    {"tpuv6e", 180.0 / 2, 180.0 / 2, 0.95, 0.85, false, 4, {4, 4, 4}, {4, 4, 4}},
    {"tpuv7e", 200.0 / 2, 200.0 / 2, 0.95, 0.95, false, 4, {4, 4, 4}, {4, 4, 4}},
    {"tpuv7el200", 2000.0 / 2, 2000.0 / 2, 0.95, 0.95, false, 4, {8, 8, 8}, {4, 4, 4}},
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
  int num_hops;              // Number of hops for communication
  double per_link_cost_us;   // Cost per link in microseconds
};

// Forward declarations
std::vector<int64_t> GetDeviceIdsFromOneReplicaGroup(const xla::HloInstruction* instr);
CommType DetermineCommType(const std::vector<int64_t>& device_ids, uint64_t replica_group_size, uint64_t intranode_size, const xla::HloInstruction* instr);
uint64_t GetNumReplicaGroups(const xla::HloInstruction* instr);
uint64_t GetReplicaGroupSize(const xla::HloInstruction* instr);
std::string GetDataTypeFromInstruction(const xla::HloInstruction* instr);
bool IsTpuDevice(const std::string& hardware_arch);



// Helper function to calculate total size from mesh shape
int CalculateMeshSize(const std::vector<int>& mesh_shape) {
  if (mesh_shape.empty()) {
    return 1;  // Default size for non-torus devices
  }
  int size = 1;
  for (int dim : mesh_shape) {
    size *= dim;
  }
  return size;
}

// Helper function to validate IntraNodeConfig values
void ValidateIntraNodeConfig(const IntraNodeConfig& config) {
  CHECK_GE(config.size, 1) << "intranode_config.size must be at least 1, got: " << config.size;
  CHECK_GE(config.bandwidth_gbps, 1.0) << "intranode_config.bandwidth_gbps must be at least 1.0, got: " << config.bandwidth_gbps;
  CHECK_GT(config.efficiency_factor, 0.0) << "intranode_config.efficiency_factor must be greater than 0.0, got: " << config.efficiency_factor;
  CHECK_LE(config.efficiency_factor, 1.0) << "intranode_config.efficiency_factor must be less than or equal to 1.0, got: " << config.efficiency_factor;
}

// Helper function to determine intra-node size and bandwidth based on hardware architecture
IntraNodeConfig GetIntraNodeConfigFromDeviceInfo(const stream_executor::DeviceDescription& gpu_device_info,
                                                 const std::string& hardware_architecture,
                                                 const std::string& fallback_device_type = "") {
  // Use hardware architecture name for device configuration lookup
  std::string device_name = hardware_architecture;

  // Handle empty hardware architecture by falling back to device description
  if (device_name.empty()) {
    device_name = gpu_device_info.name();
    // Handle undefined device names
    if (device_name == "<undefined>" || device_name.empty()) {
      device_name = fallback_device_type;
    }
  }

  // Search for exact device configuration match (case-insensitive)
  for (const auto& config : device_configs) {
    if (absl::AsciiStrToLower(device_name) == absl::AsciiStrToLower(config.name_pattern)) {
      // Calculate intranode_size from intranode_mesh_shape
      int intranode_size = CalculateMeshSize(config.intranode_mesh_shape);
      IntraNodeConfig result = {intranode_size, config.intranode_bandwidth_gbps, config.intranode_efficiency_factor, config.reassign_bandwidth};
      ValidateIntraNodeConfig(result);
      return result;
    }
  }

  // Debug: Print available device patterns for troubleshooting
  llvm::outs() << "DEBUG: Hardware architecture '" << device_name << "' not found in device_configs.\n";
  llvm::outs() << "DEBUG: Available device patterns (exact match required):\n";
  for (const auto& config : device_configs) {
    llvm::outs() << "  - '" << config.name_pattern << "'\n";
  }
  llvm::outs().flush();

  // Fail with error if no exact device configuration match is found
  CHECK(false) << "No exact device configuration match found for hardware architecture: '" << device_name
               << "'. Please add support for this device type in GetIntraNodeConfigFromDeviceInfo function.";
}

// Function to calculate communication cost based on device information and communication volume
CommCostStats CalculateCommCost(double per_device_comm_volume,
                               const stream_executor::DeviceDescription& gpu_device_info,
                               const xla::HloInstruction* instr, uint64_t replica_group_size,
                               uint64_t num_replica_groups,
                               const std::string& hardware_architecture,
                               const std::string& fallback_device_type = "") {
  // Get device-specific configuration (already validated in GetIntraNodeConfigFromDeviceInfo)
  IntraNodeConfig intranode_config = GetIntraNodeConfigFromDeviceInfo(gpu_device_info, hardware_architecture, fallback_device_type);

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

  // Calculate number of hops and per-link cost for GPU communication
  int num_hops = 1;  // Default to 1 hop for GPU communication
  double per_link_cost_us = 0.0;

  if (comm_type == CommType::IntraNode) {
    // For intra-node communication, assume 1 hop through NVLink/PCIe
    num_hops = 1;
    per_link_cost_us = comm_cost_us / num_hops;
  } else if (comm_type == CommType::InterNode) {
    // For inter-node communication, assume multiple hops through network
    num_hops = 2;  // Conservative estimate for network hops
    per_link_cost_us = comm_cost_us / num_hops;
  } else if (comm_type == CommType::Hierarchical) {
    // For hierarchical communication, combine intra and inter-node hops
    num_hops = 3;  // 1 for intra-node + 2 for inter-node
    per_link_cost_us = comm_cost_us / num_hops;
  }


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
    llvm::outs().flush();
  }

  return {
    comm_type,
    comm_cost_us,
    intranode_comm_vol_gb,
    internode_comm_vol_gb,
    intranode_comm_bw_gbps,
    internode_comm_bw_gbps,
    total_comm_vol_gb,
    num_hops,
    per_link_cost_us
  };
}

int CalculateMaxHop(const std::vector<int64_t>& device_ids, const std::vector<int>& mesh_shape) {
  float max_hop = 0;
  // Go through each dim in the mesh
  for (int i = 0; i < mesh_shape.size(); ++i) {
      max_hop += mesh_shape[i] / 4.0;
  }

  return max_hop;
}

int CalculateAverageHops(const std::vector<int>& mesh_shape) {
    double avg_hops = 0.0;
    for (int dim_size : mesh_shape) {
        avg_hops += dim_size / 4.0;
    }
    return static_cast<int>(std::ceil(avg_hops));
}

// Structure to hold torus hop results
struct TorusHops {
  int copper_hops;  // Number of hops over copper links (within pod slice)
  int ici_hops;     // Number of hops over ICI links (between pod slices)
};

// Helper function to convert node ID to 3D coordinates
std::tuple<int, int, int> IdToCoord(int64_t node_id, const std::vector<int>& dims) {
  int x = node_id % dims[0];
  int y = (node_id / dims[0]) % dims[1];
  int z = (node_id / (dims[0] * dims[1])) % dims[2];
  return std::make_tuple(x, y, z);
}

// Helper function to compute torus distance in 3D
int TorusDistance3D(const std::tuple<int, int, int>& p1, const std::tuple<int, int, int>& p2, const std::vector<int>& dims) {
  int dist = 0;
  int delta_x = std::abs(std::get<0>(p1) - std::get<0>(p2));
  int delta_y = std::abs(std::get<1>(p1) - std::get<1>(p2));
  int delta_z = std::abs(std::get<2>(p1) - std::get<2>(p2));

  dist += std::min(delta_x, dims[0] - delta_x);
  dist += std::min(delta_y, dims[1] - delta_y);
  dist += std::min(delta_z, dims[2] - delta_z);

  return dist;
}

// Calculate hops over copper and ICI links in a 3D torus topology from integer node IDs
TorusHops TorusHopsInt(int64_t src_id, int64_t dst_id, const std::vector<int>& pod_dims, const std::vector<int>& slice_dims) {
  // Total nodes in pod slice and slicing arrangement
  int64_t pod_size = pod_dims[0] * pod_dims[1] * pod_dims[2];

  // Extract pod slice IDs from node IDs
  int64_t src_pod_id = src_id / pod_size;
  int64_t dst_pod_id = dst_id / pod_size;

  // Extract slice node IDs within pod
  int64_t src_slice_id = src_id % pod_size;
  int64_t dst_slice_id = dst_id % pod_size;

  // Convert IDs to coordinates
  auto src_pod_coord = IdToCoord(src_pod_id, pod_dims);
  auto dst_pod_coord = IdToCoord(dst_pod_id, pod_dims);

  auto src_slice_coord = IdToCoord(src_slice_id, slice_dims);
  auto dst_slice_coord = IdToCoord(dst_slice_id, slice_dims);

  // Calculate hops
  int copper_hops = TorusDistance3D(src_slice_coord, dst_slice_coord, pod_dims);
  int ici_hops = TorusDistance3D(src_pod_coord, dst_pod_coord, slice_dims);

  return {copper_hops, ici_hops};
}

// Function to calculate TPU communication cost based on 3D torus topology
CommCostStats CalculateTpuCommCost(double per_device_comm_volume,
                                   const stream_executor::DeviceDescription& tpu_device_info,
                                   const xla::HloInstruction* instr, uint64_t replica_group_size,
                                   uint64_t num_replica_groups,
                                   const std::vector<int64_t>& device_ids,
                                   const std::vector<int>& mesh_shape,
                                   const std::string& hardware_architecture,
                                   const std::string& fallback_device_type = "") {
  // Validate inputs
  CHECK_GE(per_device_comm_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << per_device_comm_volume;
  CHECK_GE(replica_group_size, 1) << "replica_group_size must be at least 1, got: " << replica_group_size;
  CHECK_GE(num_replica_groups, 1) << "num_replica_groups must be at least 1, got: " << num_replica_groups;
  CHECK_EQ(mesh_shape.size(), 3) << "Mesh shape must be 3D for TPU torus";
  CHECK_EQ(device_ids.size(), replica_group_size) << "Device IDs size must match replica group size";

  // Get device-specific ICI bandwidth from device configuration
  IntraNodeConfig intranode_config = GetIntraNodeConfigFromDeviceInfo(tpu_device_info, hardware_architecture, fallback_device_type);

  // Get the device configuration to access links_per_chip, mesh shapes, bandwidths, and efficiency factors using hardware architecture
  int links_per_chip = 6;  // Default for TPU v4
  std::vector<int> intranode_mesh_shape = {4, 4, 4};  // Default intra-node mesh shape
  std::vector<int> internode_mesh_shape = {4, 4, 4};  // Default inter-node mesh shape
  double intranode_bandwidth_gbps = intranode_config.bandwidth_gbps;  // Default to intranode config
  double internode_bandwidth_gbps = 10.0;  // Default inter-node bandwidth
  double intranode_efficiency_factor = intranode_config.efficiency_factor;  // Default to intranode config
  double internode_efficiency_factor = 0.85;  // Default inter-node efficiency

  // Find the device configuration to get all parameters using hardware architecture
  for (const auto& config : device_configs) {
      if (absl::StrContains(absl::AsciiStrToLower(config.name_pattern), absl::AsciiStrToLower(hardware_architecture))) {
      links_per_chip = config.links_per_chip;
      intranode_bandwidth_gbps = config.intranode_bandwidth_gbps;
      internode_bandwidth_gbps = config.internode_bandwidth_gbps;
      intranode_efficiency_factor = config.intranode_efficiency_factor;
      internode_efficiency_factor = config.internode_efficiency_factor;
      if (!config.intranode_mesh_shape.empty()) {
        intranode_mesh_shape = config.intranode_mesh_shape;
      }
      if (!config.internode_mesh_shape.empty()) {
        internode_mesh_shape = config.internode_mesh_shape;
      }
      break;
    }
  }
  // Calculate sizes from mesh shapes
  int intranode_size = CalculateMeshSize(intranode_mesh_shape);
  int internode_size = CalculateMeshSize(internode_mesh_shape);

  // Calculate bandwidth per link for both intra-node and inter-node (already per-link unidirectional)
  double intranode_bandwidth_per_link_gbps = intranode_bandwidth_gbps * intranode_efficiency_factor;
  double internode_bandwidth_per_link_gbps = internode_bandwidth_gbps * internode_efficiency_factor;
  // Calculate communication volume per device in GB
  double per_device_comm_vol_gb = per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  // Find the bottleneck hop
  // Loop through pair of device ids
  // form pairs to the right. Last node should be paired with the first node.
  double max_comm_cost_us = 0.0;
  TorusHops max_torus_hops = {0, 0};
  int max_hop_src = 0;
  int max_hop_dst = 0;
  for (size_t i = 0; i < device_ids.size(); ++i) {
      TorusHops torus_hops = TorusHopsInt(device_ids[i], device_ids[(i + 1) % device_ids.size()], internode_mesh_shape, intranode_mesh_shape);
    int number_of_hops = torus_hops.copper_hops + torus_hops.ici_hops;
    double copper_cost_us = (per_device_comm_vol_gb / intranode_bandwidth_per_link_gbps) * 1e6 * torus_hops.copper_hops;
    int ici_hops = 0;
    if (torus_hops.ici_hops > 0) {
        ici_hops = torus_hops.ici_hops;
    }
    double ici_cost_us = (per_device_comm_vol_gb / internode_bandwidth_per_link_gbps) * 1e6 * (ici_hops);
    double comm_cost_us = copper_cost_us + ici_cost_us;
    if (comm_cost_us > max_comm_cost_us) {
      max_comm_cost_us = comm_cost_us;
      max_torus_hops = torus_hops;
      max_hop_src = device_ids[i];
      max_hop_dst = device_ids[(i + 1) % device_ids.size()];
    }
  }

  int number_of_hops = max_torus_hops.copper_hops + max_torus_hops.ici_hops;
  // For TPU, all communication is intra-pod (within the 3D torus)
  double intranode_comm_vol_gb = 0.0;
  double internode_comm_vol_gb = 0.0;
  if (max_torus_hops.copper_hops > 0){
      intranode_comm_vol_gb = per_device_comm_vol_gb;
  }
  if (max_torus_hops.ici_hops > 0) {
      internode_comm_vol_gb = per_device_comm_vol_gb;
  }

  double total_comm_vol_gb = per_device_comm_vol_gb * max_torus_hops.copper_hops + per_device_comm_vol_gb * max_torus_hops.ici_hops;

  CommType comm_type;
  if (max_torus_hops.ici_hops == 0){
      comm_type = CommType::IntraNode;
  } else {
      comm_type = CommType::InterNode;
  }

  // Debug logging
  if (std::isnan(max_comm_cost_us) || std::isnan(intranode_comm_vol_gb) || std::isnan(total_comm_vol_gb)) {
    llvm::outs() << "WARNING: NaN detected in CalculateTpuCommCost for instruction: " << instr->ToString() << "\n";
    llvm::outs() << "  per_device_comm_volume: " << per_device_comm_volume << "\n";
    llvm::outs() << "  replica_group_size: " << replica_group_size << "\n";
    llvm::outs() << "  links_per_chip: " << links_per_chip << "\n";
    llvm::outs() << "  copper_hops: " << max_torus_hops.copper_hops << "\n";
    llvm::outs() << "  ici_hops: " << max_torus_hops.ici_hops << "\n";
    llvm::outs() << "  intranode_bandwidth_per_link_gbps: " << intranode_bandwidth_per_link_gbps << "\n";
    llvm::outs() << "  internode_bandwidth_per_link_gbps: " << internode_bandwidth_per_link_gbps << "\n";
    llvm::outs() << "  comm_cost_us: " << max_comm_cost_us << "\n";
    llvm::outs().flush();
  }

  return {
    comm_type,
    max_comm_cost_us,
    intranode_comm_vol_gb,
    internode_comm_vol_gb,
    intranode_bandwidth_per_link_gbps,  // intranode_comm_bw_gbps
    internode_bandwidth_per_link_gbps,  // internode_comm_bw_gbps
    total_comm_vol_gb,
    number_of_hops,
    max_comm_cost_us / number_of_hops  // per_link_cost_us
  };
}

struct InstructionTimelineEntry {
  const xla::HloInstruction* instruction;
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

OverlapStats CalculateInstructionLevelOverlap(const std::vector<InstructionTimelineEntry>& timeline, double overlap_factor) {
  CHECK_GE(overlap_factor, 0.0) << "overlap_factor must be non-negative, got: " << overlap_factor;
  CHECK_LE(overlap_factor, 1.0) << "overlap_factor must be <= 1.0, got: " << overlap_factor;

  if (timeline.empty()) {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, overlap_factor, {}};
  }

  double original_total_time_us = 0.0;
  double total_compute_time_us = 0.0;
  double total_comm_time_us = 0.0;

  for (const auto& entry : timeline) {
    original_total_time_us += entry.duration_us;
    if (entry.is_compute_instruction) {
      total_compute_time_us += entry.duration_us;
    }
    if (entry.is_comm_instruction) {
      total_comm_time_us += entry.duration_us;
    }
  }

  if (overlap_factor == 0.0) {
    return {
      original_total_time_us,
      original_total_time_us,
      0.0,
      0.0,
      total_compute_time_us,
      total_comm_time_us,
      overlap_factor,
      timeline
    };
  }

  std::vector<InstructionTimelineEntry> overlapped_timeline = timeline;
  double current_time_us = 0.0;
  double last_compute_time_us = 0.0;
  double last_comm_time_us = 0.0;
  for (size_t i = 0; i < overlapped_timeline.size(); ++i) {
    auto& current_instruction = overlapped_timeline[i];

    if (current_instruction.start_time_us == 0.0){
      // unscheduled instruction
      current_instruction.start_time_us = current_time_us;
      current_instruction.end_time_us = current_time_us + current_instruction.duration_us;
    } else {
      // current instruction is already scheduled.
      current_time_us = current_instruction.start_time_us;
    }
    // Try to overlap with next instruction
    if (overlap_factor > 0.0) {
      size_t j = i + 1;
      if (j < overlapped_timeline.size()) {
        auto& next_instruction = overlapped_timeline[j];
        bool can_overlap = (current_instruction.is_compute_instruction && next_instruction.is_comm_instruction) ||
                          (current_instruction.is_comm_instruction && next_instruction.is_compute_instruction);
        CHECK_EQ(next_instruction.start_time_us, 0.0) << "Next instruction start time is not 0.0";
        if (can_overlap) {
          double overlap_duration = current_instruction.duration_us * overlap_factor;
          double next_start_time = current_instruction.start_time_us + (current_instruction.duration_us - overlap_duration);
          if (next_instruction.is_compute_instruction) {
            next_start_time = std::max(next_start_time, last_compute_time_us);
          }
          else if (next_instruction.is_comm_instruction) {
            next_start_time = std::max(next_start_time, last_comm_time_us);
          }
          next_instruction.start_time_us = next_start_time;
          next_instruction.end_time_us = next_start_time + next_instruction.duration_us;
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
      last_compute_time_us = current_instruction.start_time_us + current_instruction.duration_us;
    }
    else if (current_instruction.is_comm_instruction) {
      last_comm_time_us = current_instruction.start_time_us + current_instruction.duration_us;
    }
  }

  double overlapped_total_time_us = 0.0;
  for (const auto& entry : overlapped_timeline) {
    overlapped_total_time_us = std::max(overlapped_total_time_us, entry.end_time_us);
  }

  double overlap_savings_us = original_total_time_us - overlapped_total_time_us;
  double overlap_percentage = original_total_time_us > 0.0 ? (overlap_savings_us / original_total_time_us) * 100.0 : 0.0;

  return {
    original_total_time_us,
    overlapped_total_time_us,
    overlap_savings_us,
    overlap_percentage,
    total_compute_time_us,
    total_comm_time_us,
    overlap_factor,
    overlapped_timeline
  };
}

struct CliOpts {
  std::string hlo_module_file;
  std::vector<std::string> hardware_architectures;
  std::string output_dir = "stats";
  std::string mesh_shape = "4,4,4";
  std::string overlap_factor_str = "0.0";
  double overlap_factor = 0.0;
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
bool IsTpuDevice(const std::string& hardware_arch) {
  std::string lower_arch = absl::AsciiStrToLower(hardware_arch);
  return absl::StrContains(lower_arch, "tpu");
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
  double operand_size;
  double result_size;
  std::string pattern;
  std::string formula;
};

// Helper function to calculate ring-based communication volume
CommunicationVolume CalculateRingVolume(double per_step_data, uint64_t replica_group_size,
                                       const std::string& operation_name,
                                       const std::string& formula, double multiplier = 1.0,
                                       double operand_size = 0.0, double result_size = 0.0) {
  double total_steps = replica_group_size - 1;
  double per_device_volume = per_step_data * total_steps;
  double total_volume = per_device_volume * replica_group_size;

  return {
    per_device_volume * multiplier,
    total_volume * multiplier,
    operand_size,
    result_size,
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
        CommunicationVolume result = CalculateRingVolume(per_step_data, replica_group_size, "AllReduce", formula, 2.0, operand_size, result_size);
        CHECK_GT(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
        return result;
    }
    case xla::HloOpcode::kAllGather: {
        CHECK_GE(result_size, operand_size)
            << "AllGather result size must be greater than or equal to operand size. "
            << "Operand size: " << operand_size << ", Result size: " << result_size
            << " for instruction: " << instr->ToString();

        std::string formula = "operand_size * (" + std::to_string(replica_group_size) + "-1)";
        CommunicationVolume result = CalculateRingVolume(operand_size, replica_group_size, "AllGather", formula, 1.0, operand_size, result_size);
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
        CommunicationVolume result = CalculateRingVolume(result_size, replica_group_size, "ReduceScatter", formula, 1.0, operand_size, result_size);
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
        operand_size,
        result_size,
        "Point-to-point permutation",
        "operand_size * 2"
      };
      CHECK_GT(result.per_device_volume, 0.0) << "per_device_comm_volume must be greater than 0.0, got: " << result.per_device_volume;
      return result;
    }
    default:
      return {0.0, 0.0, 0.0, 0.0, "Unknown", "Unknown"};
  }
}

int main(int argc, char* argv[]) {
  llvm::errs().tie(&llvm::outs());
  CliOpts opts;
  std::string hardware_architectures_str;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo-module-file", &opts.hlo_module_file, "Filename of HloModule"),
      tsl::Flag("hardware-architectures", &hardware_architectures_str, "Comma-separated list of hardware architectures (e.g., h100_pcie,tpuv5p,tpuv6e)"),
      tsl::Flag("output-dir", &opts.output_dir, "Output directory for CSV files (default: stats)"),
      tsl::Flag("mesh-shape", &opts.mesh_shape, "3D mesh shape for TPU communication (e.g., 4,4,4)"),
      tsl::Flag("overlap-factor", &opts.overlap_factor_str, "Compute-communication overlap factor (0.0-1.0, default: 0.0)")};
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

  // Parse overlap_factor from string
  if (!opts.overlap_factor_str.empty()) {
    char* end_ptr;
    opts.overlap_factor = std::strtod(opts.overlap_factor_str.c_str(), &end_ptr);
    if (*end_ptr != '\0' || opts.overlap_factor < 0.0 || opts.overlap_factor > 1.0) {
      llvm::outs() << "Error: Overlap factor must be a valid number between 0.0 and 1.0, got: " << opts.overlap_factor_str << "\n";
      return 1;
    }
  }

  // Parse mesh_shape from comma-separated string
  std::vector<int> mesh_shape;
  if (!opts.mesh_shape.empty()) {
    std::vector<absl::string_view> mesh_parts = absl::StrSplit(opts.mesh_shape, ',');
    for (const absl::string_view& part : mesh_parts) {
      absl::string_view trimmed_part = absl::StripAsciiWhitespace(part);
      if (!trimmed_part.empty()) {
        // Check if string contains only digits (and optional leading sign)
        bool is_valid = !trimmed_part.empty();
        if (is_valid) {
          size_t start = 0;
          if (trimmed_part[0] == '-' || trimmed_part[0] == '+') {
            start = 1;
          }
          for (size_t i = start; i < trimmed_part.length(); ++i) {
            if (!std::isdigit(trimmed_part[i])) {
              is_valid = false;
              break;
            }
          }
        }

        if (!is_valid) {
          llvm::outs() << "Error: Invalid mesh shape value: " << trimmed_part << "\n";
          return 1;
        }

        int value = std::stoi(std::string(trimmed_part));
        if (value <= 0) {
          llvm::outs() << "Error: Mesh shape values must be positive integers, got: " << value << "\n";
          return 1;
        }
        mesh_shape.push_back(value);
      }
    }
  }

  // Validate mesh_shape
  if (mesh_shape.size() != 3) {
    llvm::outs() << "Error: Mesh shape must have exactly 3 dimensions, got: " << mesh_shape.size() << "\n";
    return 1;
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  CHECK(opts.hlo_module_file.empty() == false)
      << "Path to HLO module file required";


  // Validate hardware architectures
  absl::Status validation_status = ValidateHardwareArchitectures(opts.hardware_architectures);
  if (!validation_status.ok()) {
    llvm::outs() << "Error: " << validation_status.message() << "\n";
    llvm::outs().flush();
    return 1;
  }

  llvm::outs() << "Using hardware architectures: ";
  for (size_t i = 0; i < opts.hardware_architectures.size(); ++i) {
    if (i > 0) llvm::outs() << ", ";
    llvm::outs() << opts.hardware_architectures[i];
  }
  llvm::outs() << "\n";
  llvm::outs() << "Using output directory: " << opts.output_dir << "\n";
  llvm::outs() << "Using mesh shape: " << mesh_shape[0] << "x" << mesh_shape[1] << "x" << mesh_shape[2] << "\n";
  llvm::outs().flush();

  std::string prefix = "/xla";
  auto output_path = tsl::io::JoinPath("/xla", opts.output_dir);
  // Create output directory if it doesn't exist
  if (!tsl::Env::Default()->FileExists(output_path).ok()) {
    tsl::Env::Default()->CreateDir(output_path);
  }
  std::ofstream device_stats_csv =
      createCsv(tsl::io::JoinPath(output_path, "device_stats.csv"));
  auto comp_stats_csv = createCsv(tsl::io::JoinPath(output_path, "comp_stats.csv"));
  auto comm_stats_csv = createCsv(tsl::io::JoinPath(output_path, "comm_stats.csv"));
  auto overlap_stats_csv = createCsv(tsl::io::JoinPath(output_path, "overlap_stats.csv"));
  auto instruction_timeline_csv = createCsv(tsl::io::JoinPath(output_path, "instruction_timeline.csv"));
  if (!device_stats_csv.is_open() || !comp_stats_csv.is_open() || !comm_stats_csv.is_open() ||
      !overlap_stats_csv.is_open() || !instruction_timeline_csv.is_open()) {
    return -1;
  }

  std::vector<std::vector<std::string>> device_stats_data, comp_stats_data, comm_stats_data, overlap_stats_data, instruction_timeline_data;
  std::vector<std::string> device_stats_header = {
      "device_name", "overlapped_latency_secs", "original_latency_secs", "compute_time_secs",
      "comm_time_secs", "overlap_factor", "overlap_savings_secs", "overlap_percentage", "total_instructions"};
  std::vector<std::string> comp_stats_header = {
      "comp_id",   "idx",       "inst",       "group",         "latency(µs)",
      "tflops",    "bytes_read_gb", "bytes_written_gb", "compute_time",
      "read_time", "write_time", "throughput_tflops_per_sec", "device_name", "datatype"};
  std::vector<std::string> comm_stats_header = {
      "comp_id",   "instruction_name", "opcode", "device_name", "comm_type",
      "comm_cost_us", "replica_group_size", "num_replica_groups",
      "intranode_comm_vol_gb", "internode_comm_vol_gb", "intranode_comm_bw_gbps", "internode_comm_bw_gbps",
      "total_comm_vol_gb", "operand_size_bytes", "result_size_bytes", "datatype", "num_hops", "per_link_cost_us"};
  std::vector<std::string> overlap_stats_header = {
      "device_name", "overlap_factor", "original_total_time_secs", "overlapped_total_time_secs",
      "total_compute_time_secs", "total_comm_time_secs", "overlap_savings_secs", "overlap_percentage", "total_instructions"};
  std::vector<std::string> instruction_timeline_header = {
      "device_name", "instruction_name", "instruction_type", "original_duration_us",
      "overlapped_start_time_us", "overlapped_end_time_us", "overlapped_duration_us", "is_compute", "is_comm"};

  device_stats_data.push_back(device_stats_header);
  comp_stats_data.push_back(comp_stats_header);
  comm_stats_data.push_back(comm_stats_header);
  overlap_stats_data.push_back(overlap_stats_header);
  instruction_timeline_data.push_back(instruction_timeline_header);
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
    gpu_device_info.set_name(device_name);
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
    std::vector<InstructionTimelineEntry> instruction_timeline;
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
            if (IsTpuDevice(spec_file_name)) {
              // Get device IDs and mesh shape for TPU communication cost calculation
              auto device_ids = GetDeviceIdsFromOneReplicaGroup(instr);
              // Use mesh shape from command line argument
              comm_stats = CalculateTpuCommCost(comm_volume.per_device_volume, gpu_device_info, instr, replica_group_size, num_replica_groups, device_ids, mesh_shape, spec_file_name, "");
            } else {
              comm_stats = CalculateCommCost(comm_volume.per_device_volume, gpu_device_info, instr, replica_group_size, num_replica_groups, spec_file_name, "");
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
                std::to_string(comm_volume.operand_size),
                std::to_string(comm_volume.result_size),
                GetDataTypeFromInstruction(instr),
                std::to_string(comm_stats.num_hops),
                std::to_string(comm_stats.per_link_cost_us)
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

        // Add to instruction timeline if cost > 0
        if (cost > 0.0) {
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
            case xla::HloOpcode::kWhile: {
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

          instruction_timeline.push_back(timeline_entry);
        }

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

    // Calculate instruction-level overlapped latency
    OverlapStats overlap_stats = CalculateInstructionLevelOverlap(instruction_timeline, opts.overlap_factor);

    // Print entry computation timing breakdown
    llvm::outs() << "=== Entry Computation Timing Breakdown ===\n";
    llvm::outs() << "Device: " << device_name << "\n";
    llvm::outs() << "Total Instructions Processed: " << instruction_timeline.size() << "\n";
    if (opts.overlap_factor > 0.0) {
      llvm::outs() << "Overlap Factor: " << opts.overlap_factor << "\n";
      llvm::outs() << "Original Total Time: " << overlap_stats.original_total_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Overlapped Total Time: " << overlap_stats.overlapped_total_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Overlap Savings: " << overlap_stats.overlap_savings_us / 1e6 << " seconds ("
                   << overlap_stats.overlap_percentage << "%)\n";
    } else {
      llvm::outs() << "Total Compute Time: " << overlap_stats.total_compute_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Total Communication Time: " << overlap_stats.total_comm_time_us / 1e6 << " seconds\n";
      llvm::outs() << "Total Time: " << overlap_stats.original_total_time_us / 1e6 << " seconds\n";
    }
    llvm::outs() << "==========================================\n";
    llvm::outs().flush();

    // Use overlapped time for device_stats
    auto overlapped_comp_cost_in_secs = overlap_stats.overlapped_total_time_us / 1e6;
    auto original_comp_cost_in_secs = overlap_stats.original_total_time_us / 1e6;
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
         std::to_string(instruction_timeline.size())});

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
        std::to_string(instruction_timeline.size())
    };
    overlap_stats_data.push_back(overlap_row);

    // Add instruction timeline to CSV data
    for (const auto& entry : overlap_stats.timeline) {
      std::string instruction_type = "other";
      if (entry.is_compute_instruction) instruction_type = "compute";
      if (entry.is_comm_instruction) instruction_type = "comm";

      std::vector<std::string> timeline_row = {
          device_name,
          entry.instruction_name,
          instruction_type,
          std::to_string(entry.duration_us),
          std::to_string(entry.start_time_us),
          std::to_string(entry.end_time_us),
          std::to_string(entry.end_time_us - entry.start_time_us),
          entry.is_compute_instruction ? "true" : "false",
          entry.is_comm_instruction ? "true" : "false"
      };
      instruction_timeline_data.push_back(timeline_row);
    }

    llvm::outs().flush();
  }
  writeCsv(device_stats_csv, device_stats_data);
  writeCsv(comp_stats_csv, comp_stats_data);
  writeCsv(comm_stats_csv, comm_stats_data);
  writeCsv(overlap_stats_csv, overlap_stats_data);
  writeCsv(instruction_timeline_csv, instruction_timeline_data);
  device_stats_csv.close();
  comp_stats_csv.close();
  comm_stats_csv.close();
  overlap_stats_csv.close();
  instruction_timeline_csv.close();
  llvm::outs() << "CSV files saved to: " << tsl::io::JoinPath(output_path, "device_stats.csv")
               << ", " << tsl::io::JoinPath(output_path, "comp_stats.csv")
               << ", " << tsl::io::JoinPath(output_path, "comm_stats.csv")
               << ", " << tsl::io::JoinPath(output_path, "overlap_stats.csv")
               << ", " << tsl::io::JoinPath(output_path, "instruction_timeline.csv") << "\n";
  llvm::outs() << "Done\n";
  return 0;
}
