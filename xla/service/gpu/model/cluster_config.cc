#include "cluster_config.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

// TpuClusterConfig Implementation
TpuClusterConfig::TpuClusterConfig()
    : name_pattern_(""), intranode_bandwidth_gbps_(0.0),
      internode_bandwidth_gbps_(0.0), intranode_efficiency_factor_(0.0),
      internode_efficiency_factor_(0.0), reassign_bandwidth_(false),
      intranode_mesh_shape_(), internode_mesh_shape_(), links_per_chip_(0) {}

std::string TpuClusterConfig::GetNamePattern() const { return name_pattern_; }

double TpuClusterConfig::GetIntranodeBandwidthGbps() const {
  return intranode_bandwidth_gbps_;
}

double TpuClusterConfig::GetInternodeBandwidthGbps() const {
  return internode_bandwidth_gbps_;
}

double TpuClusterConfig::GetIntranodeEfficiencyFactor() const {
  return intranode_efficiency_factor_;
}

double TpuClusterConfig::GetInternodeEfficiencyFactor() const {
  return internode_efficiency_factor_;
}

bool TpuClusterConfig::GetReassignBandwidth() const {
  return reassign_bandwidth_;
}

std::vector<int> TpuClusterConfig::GetIntranodeMeshShape() const {
  return intranode_mesh_shape_;
}

std::vector<int> TpuClusterConfig::GetInternodeMeshShape() const {
  return internode_mesh_shape_;
}

int TpuClusterConfig::GetLinksPerChip() const { return links_per_chip_; }

void TpuClusterConfig::SetConfig(const std::string &name_pattern,
                                 double intranode_bandwidth_gbps,
                                 double internode_bandwidth_gbps,
                                 double intranode_efficiency_factor,
                                 double internode_efficiency_factor,
                                 bool reassign_bandwidth,
                                 const std::vector<int> &intranode_mesh_shape,
                                 const std::vector<int> &internode_mesh_shape) {
  name_pattern_ = name_pattern;
  intranode_bandwidth_gbps_ = intranode_bandwidth_gbps;
  internode_bandwidth_gbps_ = internode_bandwidth_gbps;
  intranode_efficiency_factor_ = intranode_efficiency_factor;
  internode_efficiency_factor_ = internode_efficiency_factor;
  reassign_bandwidth_ = reassign_bandwidth;
  intranode_mesh_shape_ = intranode_mesh_shape;
  internode_mesh_shape_ = internode_mesh_shape;
}

bool TpuClusterConfig::LoadFromFile(const std::string &config_file_path) {
  std::ifstream file(config_file_path);
  if (!file.is_open()) {
    std::cerr << "ERROR: TpuClusterConfig::LoadFromFile - Cannot open file: "
              << config_file_path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse key=value pairs
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);

    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    // Parse configuration values
    if (key == "name_pattern") {
      name_pattern_ = value;
    } else if (key == "intranode_bandwidth_gbps") {
      intranode_bandwidth_gbps_ = std::stod(value);
    } else if (key == "internode_bandwidth_gbps") {
      internode_bandwidth_gbps_ = std::stod(value);
    } else if (key == "intranode_efficiency_factor") {
      intranode_efficiency_factor_ = std::stod(value);
    } else if (key == "internode_efficiency_factor") {
      internode_efficiency_factor_ = std::stod(value);
    } else if (key == "reassign_bandwidth") {
      reassign_bandwidth_ = (value == "true" || value == "1");
    } else if (key == "intranode_mesh_shape") {
      // Parse comma-separated values like "4,4,4"
      std::istringstream iss(value);
      std::string token;
      intranode_mesh_shape_.clear();
      while (std::getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
          intranode_mesh_shape_.push_back(std::stoi(token));
        }
      }
    } else if (key == "internode_mesh_shape") {
      // Parse comma-separated values like "4,4,4"
      std::istringstream iss(value);
      std::string token;
      internode_mesh_shape_.clear();
      while (std::getline(iss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (!token.empty()) {
          internode_mesh_shape_.push_back(std::stoi(token));
        }
      }
    } else if (key == "links_per_chip") {
      links_per_chip_ = std::stoi(value);
    }
  }

  file.close();
  return true;
}

// GpuClusterConfig Implementation
GpuClusterConfig::GpuClusterConfig()
    : name_pattern_(""), intranode_efficiency_factor_(0.0),
      internode_efficiency_factor_(0.0), reassign_bandwidth_(false),
      scalable_unit_count_(0), compute_rack_count_(0),
      compute_trays_per_rack_(0), compute_units_per_tray_(0), nic_per_tray_(0),
      nic_speed_gbytes_(0.0), scaleup_fabric_link_bw_gbytes_(0.0),
      scaleup_fabric_port_count_(0), scaleup_switch_port_count_(0),
      scaleup_switch_latency_(0), scaleup_fabric_topology_(""),
      pcie_link_bw_(0.0), pcie_link_count_(0), scaleout_fabric_topology_(""),
      mgmt_rack_count_(0), leaf_switch_port_count_(0),
      spine_switch_port_count_(0), core_switch_port_count_(0),
      scaleout_subscription_(""), scaleup_domain_({0, 0, {}, "", 0.0}),
      scaleout_domain_({0, 0, 0, 0.0, {}, {}, {}, 0, 0, 0, 0, {}, ""}) {}

std::string GpuClusterConfig::GetNamePattern() const { return name_pattern_; }

double GpuClusterConfig::GetIntranodeEfficiencyFactor() const {
  return intranode_efficiency_factor_;
}

double GpuClusterConfig::GetInternodeEfficiencyFactor() const {
  return internode_efficiency_factor_;
}

bool GpuClusterConfig::GetReassignBandwidth() const {
  return reassign_bandwidth_;
}

const ScaleupDomain &GpuClusterConfig::GetScaleupDomain() const {
  return scaleup_domain_;
}

const ScaleoutDomain &GpuClusterConfig::GetScaleoutDomain() const {
  return scaleout_domain_;
}

void GpuClusterConfig::BuildDomains() {
  // Build scaleup domain
  double per_xpu_bw_gbytes =
      scaleup_fabric_link_bw_gbytes_ * scaleup_fabric_port_count_;
  if (scaleup_fabric_topology_ == "all2all") {
    // Scaleup domain size = number of GPUs per scaleup domain
    // Each scalable unit has compute_units_per_tray GPUs
    // Scaleup domain spans across all scalable units in the same domain
    scaleup_domain_.scaleup_device_count =
        scaleup_switch_port_count_; // 1 link per device
    scaleup_domain_.scaleup_switch_count =
        scaleup_fabric_port_count_; // 1 switch per XPU port
    scaleup_domain_.scaleup_fabric_link_bw = {
        scaleup_fabric_port_count_,
        static_cast<int>(scaleup_fabric_link_bw_gbytes_)};
    scaleup_domain_.scaleup_fabric_topology = scaleup_fabric_topology_;
    scaleup_domain_.per_xpu_bw_gbytes = per_xpu_bw_gbytes;
  } else {
    // Fail for unsupported topologies instead of using default values
    LOG(FATAL) << "GpuClusterConfig::BuildDomains - Unsupported scaleup_fabric_topology: "
               << scaleup_fabric_topology_ << ". Supported topologies: all2all";
  }

  // Build scaleout domain (fat tree)
  if (scaleout_fabric_topology_ == "rail") {
    // Parse subscription ratio
    std::vector<int> subscription_ratio;
    std::stringstream ss(scaleout_subscription_);
    std::string token;
    while (std::getline(ss, token, ':')) {
      subscription_ratio.push_back(std::stoi(token));
    }

    // Calculate leaf switches
    int leaf_switches = compute_rack_count_ * nic_per_tray_;

    // Determine rail switch type
    bool is_single_rail = (scalable_unit_count_ == 1);

    if (is_single_rail) {
      // Single rail configuration
      scaleout_domain_.leaf_switches_per_rail = leaf_switches;
      scaleout_domain_.spine_switches_per_slg = 0;
      scaleout_domain_.core_switches_per_cg = 0;
      scaleout_domain_.pcie_bw = pcie_link_bw_ * pcie_link_count_;
      scaleout_domain_.nic_to_leaf_switch_bw = {
          1, static_cast<int>(nic_speed_gbytes_)};
      scaleout_domain_.leaf_to_spine_bw = {0, 0};
      scaleout_domain_.spine_to_core_bw = {0, 0};
    } else {
      // SLG (Scalable Leaf Group) configuration
      // Calculate spine switches per SLG
      int spine_switch_count = (leaf_switches / nic_per_tray_) *
                               compute_trays_per_rack_ /
                               spine_switch_port_count_;
      if (spine_switch_count == 0)
        spine_switch_count = 1;

      // Calculate core switches per core group
      int core_switch_count = spine_switch_count * nic_per_tray_ *
                              scalable_unit_count_ / core_switch_port_count_;
      if (core_switch_count == 0)
        core_switch_count = 1;

      scaleout_domain_.leaf_switches_per_rail = leaf_switches / nic_per_tray_;
      scaleout_domain_.spine_switches_per_slg = spine_switch_count;
      scaleout_domain_.core_switches_per_cg = core_switch_count;
      scaleout_domain_.pcie_bw = pcie_link_bw_ * pcie_link_count_;
      scaleout_domain_.nic_to_leaf_switch_bw = {
          1, static_cast<int>(nic_speed_gbytes_)};
      scaleout_domain_.leaf_to_spine_bw = {spine_switch_port_count_,
                                           static_cast<int>(nic_speed_gbytes_)};
      scaleout_domain_.spine_to_core_bw = {core_switch_port_count_,
                                           static_cast<int>(nic_speed_gbytes_)};
    }

    scaleout_domain_.scalable_unit_count = scalable_unit_count_;
    scaleout_domain_.rack_count = compute_rack_count_;
    scaleout_domain_.tray_count = compute_trays_per_rack_;
    scaleout_domain_.unit_count = compute_units_per_tray_;
    scaleout_domain_.subscription_ratio = subscription_ratio;
    scaleout_domain_.fat_tree_type = "superpod";
  } else {
    // Fail for unsupported topologies instead of using default values
    LOG(FATAL) << "GpuClusterConfig::BuildDomains - Unsupported scaleout_fabric_topology: "
               << scaleout_fabric_topology_ << ". Supported topologies: rail";
  }
}

std::pair<int, int>
GpuClusterConfig::GpuIdToScaleupCoordinates(int gpu_id) const {
  // gpu_id is the global gpu id
  // scaleup_domain_size is the first dimension in the coordinate
  int scaleup_domain_size = scaleup_domain_.scaleup_device_count;
  int scaleup_domain_id = gpu_id / scaleup_domain_size;
  int scaleup_domain_pos = gpu_id % scaleup_domain_size;
  return {scaleup_domain_id, scaleup_domain_pos};
}

std::tuple<int, int, int, int>
GpuClusterConfig::GpuIdToScaleoutCoordinates(int gpu_id) const {
  // Extract domain dimensions
  int unit_count = scaleout_domain_.unit_count; // Units per tray
  int tray_count = scaleout_domain_.tray_count; // Trays per rack
  int rack_count = scaleout_domain_.rack_count; // Racks per scalable unit
  int scalable_unit_count =
      scaleout_domain_.scalable_unit_count; // Total scalable units

  // Calculate sizes
  int tray_size = unit_count;                      // Units per tray
  int rack_size = tray_count * tray_size;          // Units per rack
  int scalable_unit_size = rack_count * rack_size; // Units per scalable unit

  // Find scalable unit
  int scalable_unit_id = gpu_id / scalable_unit_size;
  if (scalable_unit_id >= scalable_unit_count) {
    scalable_unit_id = scalable_unit_count - 1; // Clamp to valid range
  }

  // Offset within the scalable unit
  int offset_within_unit = gpu_id % scalable_unit_size;

  // Find rack within scalable unit
  int rack_id = offset_within_unit / rack_size;

  // Offset within the rack
  int offset_within_rack = offset_within_unit % rack_size;

  // Find tray within rack
  int tray_id = offset_within_rack / tray_size;

  // Unit within tray
  int unit_id = offset_within_rack % tray_size;

  return {scalable_unit_id, rack_id, tray_id, unit_id};
}

std::vector<PathComponent>
GpuClusterConfig::PathBetweenDevices(int src_device_id,
                                     int dest_device_id) const {
  auto scaleup_src_coord = GpuIdToScaleupCoordinates(src_device_id);
  auto scaleup_dest_coord = GpuIdToScaleupCoordinates(dest_device_id);
  auto scaleout_src_coord = GpuIdToScaleoutCoordinates(src_device_id);
  auto scaleout_dest_coord = GpuIdToScaleoutCoordinates(dest_device_id);

  int scaleup_domain_size = scaleup_domain_.scaleup_device_count;

  // Check if this is a B200, R200, RCPX, B300, or Calcium (q250) configuration
  // (based on name pattern). q250 uses the 2-level stopgap path until Step 7
  // introduces CalciumClusterConfig with proper 4-level fabric routing.
  if (name_pattern_.find("b200") != std::string::npos ||
      name_pattern_.find("r200") != std::string::npos ||
      name_pattern_.find("r576") != std::string::npos ||
      name_pattern_.find("rcpx") != std::string::npos ||
      name_pattern_.find("b300") != std::string::npos ||
      name_pattern_.find("q250") != std::string::npos ||
      name_pattern_.find("calcium") != std::string::npos) {
    if (scaleup_src_coord.first == scaleup_dest_coord.first) {
      // Same scaleup domain - direct NvSwitch connection
      return {PathComponent::GPU, PathComponent::NvSwitch, PathComponent::GPU};
    } else {
      // Different scaleup domains
      int src_scalable_unit = std::get<0>(scaleout_src_coord);
      int dest_scalable_unit = std::get<0>(scaleout_dest_coord);
      int src_rack = std::get<1>(scaleout_src_coord);
      int dest_rack = std::get<1>(scaleout_dest_coord);

      // Assert that either their rack id or scalable unit id is different
      if (src_rack == dest_rack && src_scalable_unit == dest_scalable_unit) {
        // Same scalable unit and same rack - should use scaleup
        return {PathComponent::GPU, PathComponent::NvSwitch,
                PathComponent::GPU};
      }

      if (src_scalable_unit == dest_scalable_unit) {
        // Same scalable unit, different racks
        int src_offset = src_device_id % scaleup_domain_size;
        int dest_offset = dest_device_id % scaleup_domain_size;
        if (src_offset == dest_offset) {
          // Same offset - direct rail switch connection
          return {PathComponent::GPU, PathComponent::NIC,
                  PathComponent::RailSwitch, PathComponent::NIC,
                  PathComponent::GPU};
        } else {
          // Different offset - need to go through NvSwitch first
          return {PathComponent::GPU,        PathComponent::NvSwitch,
                  PathComponent::GPU,        PathComponent::NIC,
                  PathComponent::RailSwitch, PathComponent::NIC,
                  PathComponent::GPU};
        }
      } else {
        // Different scalable units - full scaleout path
        return {PathComponent::GPU,        PathComponent::NIC,
                PathComponent::RailSwitch, PathComponent::SpineSwitch,
                PathComponent::CoreSwitch, PathComponent::SpineSwitch,
                PathComponent::RailSwitch, PathComponent::NIC,
                PathComponent::GPU};
      }
    }
  } else {
    // Default path for unsupported configurations
    return {PathComponent::GPU, PathComponent::NIC, PathComponent::GPU};
  }
}

std::vector<double> GpuClusterConfig::PathToBandwidth(
    const std::vector<PathComponent> &path) const {
  std::vector<double> bandwidths;

  // Calculate bandwidth values from domain configurations
  double scaleup_fabric_link_bw = scaleup_domain_.scaleup_fabric_link_bw[0] *
                                  scaleup_domain_.scaleup_fabric_link_bw[1];
  double pcie_bw = scaleout_domain_.pcie_bw;
  double nic_bw = scaleout_domain_.nic_to_leaf_switch_bw[0] *
                  scaleout_domain_.nic_to_leaf_switch_bw[1];

  for (size_t i = 0; i < path.size() - 1; ++i) {
    PathComponent src_comp = path[i];
    PathComponent dest_comp = path[i + 1];

    if ((src_comp == PathComponent::GPU &&
         dest_comp == PathComponent::NvSwitch) ||
        (src_comp == PathComponent::NvSwitch &&
         dest_comp == PathComponent::GPU)) {
      bandwidths.push_back(scaleup_fabric_link_bw);
    } else if ((src_comp == PathComponent::GPU &&
                dest_comp == PathComponent::NIC) ||
               (src_comp == PathComponent::NIC &&
                dest_comp == PathComponent::GPU)) {
      bandwidths.push_back(pcie_bw);
    } else if ((src_comp == PathComponent::NIC &&
                dest_comp == PathComponent::RailSwitch) ||
               (src_comp == PathComponent::RailSwitch &&
                dest_comp == PathComponent::NIC) ||
               (src_comp == PathComponent::RailSwitch &&
                dest_comp == PathComponent::SpineSwitch) ||
               (src_comp == PathComponent::SpineSwitch &&
                dest_comp == PathComponent::RailSwitch) ||
               (src_comp == PathComponent::SpineSwitch &&
                dest_comp == PathComponent::CoreSwitch) ||
               (src_comp == PathComponent::CoreSwitch &&
                dest_comp == PathComponent::SpineSwitch)) {
      bandwidths.push_back(nic_bw);
    } else {
      // Default bandwidth for unknown path components
      bandwidths.push_back(1.0);
    }
  }

  return bandwidths;
}

CommType GpuClusterConfig::DetermineCommTypeFromPath(
    const std::vector<PathComponent>& path) const {
  // Check what components are present in the path
  bool has_gpu = false;
  bool has_nvswitch = false;
  bool has_nic = false;
  bool has_rail = false;
  bool has_spine = false;
  bool has_core = false;

  for (const auto& component : path) {
    switch (component) {
      case PathComponent::GPU:
        has_gpu = true;
        break;
      case PathComponent::NvSwitch:
        has_nvswitch = true;
        break;
      case PathComponent::NIC:
        has_nic = true;
        break;
      case PathComponent::RailSwitch:
        has_rail = true;
        break;
      case PathComponent::SpineSwitch:
        has_spine = true;
        break;
      case PathComponent::CoreSwitch:
        has_core = true;
        break;
      // Calcium-only path components. GpuClusterConfig::PathBetweenDevices
      // never emits these; the cases exist purely to silence -Wswitch.
      case PathComponent::L1Switch:
      case PathComponent::L2Switch:
      case PathComponent::L3Switch:
      case PathComponent::ToRSwitch:
      case PathComponent::OpticalSwitch:
      case PathComponent::OpticalSpine:
      case PathComponent::EthSwitch:
        break;
    }
  }

  // Apply the rules based on path components
  if (has_gpu && has_nvswitch && !has_nic && !has_rail && !has_spine && !has_core) {
    return CommType::ScaleUp;
  } else if (has_gpu && has_nic && has_rail && !has_nvswitch && !has_spine && !has_core) {
    return CommType::Rail;
  } else if (has_gpu && has_nvswitch && has_rail && has_nic && !has_spine && !has_core) {
    return CommType::RailOffset;
  } else if (has_gpu && has_nic && has_rail && has_spine && has_core) {
    return CommType::ScaleOut;
  } else {
    // Fail fatally if CommType could not be determined
    LOG(FATAL) << "GpuClusterConfig::DetermineCommTypeFromPath - Cannot determine CommType from path components. "
               << "Path contains: GPU=" << has_gpu << ", NvSwitch=" << has_nvswitch
               << ", NIC=" << has_nic << ", Rail=" << has_rail
               << ", Spine=" << has_spine << ", Core=" << has_core;
  }
}

std::pair<int, int> GpuClusterConfig::CountHopsByDomain(
    const std::vector<PathComponent>& path) const {
  int scaleup_hops = 0;
  int scaleout_hops = 0;

  // Count hops between consecutive components in the path
  for (size_t i = 0; i < path.size() - 1; ++i) {
    PathComponent src = path[i];
    PathComponent dest = path[i + 1];

    // Determine if this hop is in scale-up domain or scale-out domain
    bool is_scaleup_hop = false;
    bool is_scaleout_hop = false;

    // Scale-up domain hops: GPU <-> NvSwitch
    if ((src == PathComponent::GPU && dest == PathComponent::NvSwitch) ||
        (src == PathComponent::NvSwitch && dest == PathComponent::GPU)) {
      is_scaleup_hop = true;
    }
    // Scale-out domain hops: all other combinations
    else if ((src == PathComponent::GPU && dest == PathComponent::NIC) ||
             (src == PathComponent::NIC && dest == PathComponent::GPU) ||
             (src == PathComponent::NIC && dest == PathComponent::RailSwitch) ||
             (src == PathComponent::RailSwitch && dest == PathComponent::NIC) ||
             (src == PathComponent::RailSwitch && dest == PathComponent::SpineSwitch) ||
             (src == PathComponent::SpineSwitch && dest == PathComponent::RailSwitch) ||
             (src == PathComponent::SpineSwitch && dest == PathComponent::CoreSwitch) ||
             (src == PathComponent::CoreSwitch && dest == PathComponent::SpineSwitch)) {
      is_scaleout_hop = true;
    }

    if (is_scaleup_hop) {
      scaleup_hops++;
    } else if (is_scaleout_hop) {
      scaleout_hops++;
    }
  }

  return {scaleup_hops, scaleout_hops};
}

void GpuClusterConfig::SetConfig(
    const std::string &name_pattern, double intranode_efficiency_factor,
    double internode_efficiency_factor, bool reassign_bandwidth,
    int scalable_unit_count, int compute_rack_count, int compute_trays_per_rack,
    int compute_units_per_tray, int nic_per_tray, double nic_speed_gbytes,
    double scaleup_fabric_link_bw_gbytes, int scaleup_fabric_port_count,
    int scaleup_switch_port_count, int scaleup_switch_latency,
    const std::string &scaleup_fabric_topology, double pcie_link_bw,
    int pcie_link_count, const std::string &scaleout_fabric_topology,
    int mgmt_rack_count, int leaf_switch_port_count,
    int spine_switch_port_count, int core_switch_port_count,
    const std::string &scaleout_subscription) {
  name_pattern_ = name_pattern;
  intranode_efficiency_factor_ = intranode_efficiency_factor;
  internode_efficiency_factor_ = internode_efficiency_factor;
  reassign_bandwidth_ = reassign_bandwidth;
  scalable_unit_count_ = scalable_unit_count;
  compute_rack_count_ = compute_rack_count;
  compute_trays_per_rack_ = compute_trays_per_rack;
  compute_units_per_tray_ = compute_units_per_tray;
  nic_per_tray_ = nic_per_tray;
  nic_speed_gbytes_ = nic_speed_gbytes;
  scaleup_fabric_link_bw_gbytes_ = scaleup_fabric_link_bw_gbytes;
  scaleup_fabric_port_count_ = scaleup_fabric_port_count;
  scaleup_switch_port_count_ = scaleup_switch_port_count;
  scaleup_switch_latency_ = scaleup_switch_latency;
  scaleup_fabric_topology_ = scaleup_fabric_topology;
  pcie_link_bw_ = pcie_link_bw;
  pcie_link_count_ = pcie_link_count;
  scaleout_fabric_topology_ = scaleout_fabric_topology;
  mgmt_rack_count_ = mgmt_rack_count;
  leaf_switch_port_count_ = leaf_switch_port_count;
  spine_switch_port_count_ = spine_switch_port_count;
  core_switch_port_count_ = core_switch_port_count;
  scaleout_subscription_ = scaleout_subscription;

  // Build domain configurations after setting all parameters
  BuildDomains();
}

bool GpuClusterConfig::LoadFromFile(const std::string &config_file_path) {
  std::ifstream file(config_file_path);
  if (!file.is_open()) {
    std::cerr << "ERROR: GpuClusterConfig::LoadFromFile - Cannot open file: "
              << config_file_path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Parse key=value pairs
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);

    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    // Parse configuration values
    if (key == "name_pattern") {
      name_pattern_ = value;
    } else if (key == "intranode_efficiency_factor") {
      intranode_efficiency_factor_ = std::stod(value);
    } else if (key == "internode_efficiency_factor") {
      internode_efficiency_factor_ = std::stod(value);
    } else if (key == "reassign_bandwidth") {
      reassign_bandwidth_ = (value == "true" || value == "1");
    } else if (key == "scalable_unit_count") {
      scalable_unit_count_ = std::stoi(value);
    } else if (key == "compute_rack_count") {
      compute_rack_count_ = std::stoi(value);
    } else if (key == "compute_trays_per_rack") {
      compute_trays_per_rack_ = std::stoi(value);
    } else if (key == "compute_units_per_tray") {
      compute_units_per_tray_ = std::stoi(value);
    } else if (key == "nic_per_tray") {
      nic_per_tray_ = std::stoi(value);
    } else if (key == "nic_speed_gbytes") {
      nic_speed_gbytes_ = std::stod(value);
    } else if (key == "scaleup_fabric_link_bw_gbytes") {
      scaleup_fabric_link_bw_gbytes_ = std::stod(value);
    } else if (key == "scaleup_fabric_port_count") {
      scaleup_fabric_port_count_ = std::stoi(value);
    } else if (key == "scaleup_switch_port_count") {
      scaleup_switch_port_count_ = std::stoi(value);
    } else if (key == "scaleup_switch_latency") {
      scaleup_switch_latency_ = std::stoi(value);
    } else if (key == "scaleup_fabric_topology") {
      scaleup_fabric_topology_ = value;
    } else if (key == "pcie_link_bw") {
      pcie_link_bw_ = std::stod(value);
    } else if (key == "pcie_link_count") {
      pcie_link_count_ = std::stoi(value);
    } else if (key == "scaleout_fabric_topology") {
      scaleout_fabric_topology_ = value;
    } else if (key == "mgmt_rack_count") {
      mgmt_rack_count_ = std::stoi(value);
    } else if (key == "leaf_switch_port_count") {
      leaf_switch_port_count_ = std::stoi(value);
    } else if (key == "spine_switch_port_count") {
      spine_switch_port_count_ = std::stoi(value);
    } else if (key == "core_switch_port_count") {
      core_switch_port_count_ = std::stoi(value);
    } else if (key == "scaleout_subscription") {
      scaleout_subscription_ = value;
    }
  }

  file.close();

  // Build domain configurations after loading all parameters
  BuildDomains();

  return true;
}

// Factory function to create device config based on device type
std::unique_ptr<ClusterConfig>
CreateClusterConfig(const std::string &device_type) {
  if (device_type.find("tpu") != std::string::npos) {
    return std::make_unique<TpuClusterConfig>();
  } else if (device_type.find("q300") != std::string::npos ||
             device_type.find("q302") != std::string::npos ||
             device_type.find("q350") != std::string::npos) {
    // q302 is a 3-rack deployment variant of q300; q350 uses the same ESUN
    // flat-mesh cluster class with different socs_per_card / cards_per_rack.
    return std::make_unique<Q300ClusterConfig>();
  } else if (device_type.find("q250") != std::string::npos ||
             device_type.find("calcium") != std::string::npos) {
    // Calcium uses a dedicated 4-level fabric-routing config that
    // explicitly models L1/L2/L3 PCIe + L4 Ethernet RoCE v2 with
    // oversubscription and replica-group contention.
    return std::make_unique<CalciumClusterConfig>();
  } else if (device_type.find("b200") != std::string::npos ||
             device_type.find("r200") != std::string::npos ||
             device_type.find("r576") != std::string::npos ||
             device_type.find("rcpx") != std::string::npos ||
             device_type.find("b300") != std::string::npos) {
    return std::make_unique<GpuClusterConfig>();
  } else {
    std::cerr << "ERROR: CreateClusterConfig - Unknown device type: "
              << device_type << std::endl;
    return nullptr;
  }
}

// Global cache for loaded configs to avoid reloading from disk
static std::unordered_map<std::string, std::unique_ptr<ClusterConfig>>
    config_cache;

// Helper function to get device config by name pattern
std::unique_ptr<ClusterConfig> GetClusterConfigByName(
    const std::string &device_name, const std::string &configs_dir) {
  // Check cache first
  auto it = config_cache.find(device_name);
  if (it != config_cache.end()) {
    // Return a copy of the cached config
    if (device_name.find("tpu") != std::string::npos) {
      return std::make_unique<TpuClusterConfig>(
          *static_cast<TpuClusterConfig *>(it->second.get()));
    } else if (device_name.find("q300") != std::string::npos ||
               device_name.find("q302") != std::string::npos ||
               device_name.find("q350") != std::string::npos) {
      return std::make_unique<Q300ClusterConfig>(
          *static_cast<Q300ClusterConfig *>(it->second.get()));
    } else if (device_name.find("q250") != std::string::npos ||
               device_name.find("calcium") != std::string::npos) {
      return std::make_unique<CalciumClusterConfig>(
          *static_cast<CalciumClusterConfig *>(it->second.get()));
    } else {
      return std::make_unique<GpuClusterConfig>(
          *static_cast<GpuClusterConfig *>(it->second.get()));
    }
  }

  std::string config_file;
  if (!configs_dir.empty()) {
    config_file = tsl::io::JoinPath(configs_dir, device_name + ".config");
  } else {
    config_file = "xla/service/gpu/model/configs/" + device_name + ".config";
  }

  auto config = CreateClusterConfig(device_name);
  if (!config) {
    std::cerr << "ERROR: GetClusterConfigByName - Failed to create config for "
                 "device: "
              << device_name << std::endl;
    return nullptr;
  }

  if (config->LoadFromFile(config_file)) {
    // Cache the config for future use
    if (device_name.find("tpu") != std::string::npos) {
      config_cache[device_name] = std::make_unique<TpuClusterConfig>(
          *static_cast<TpuClusterConfig *>(config.get()));
    } else if (device_name.find("q300") != std::string::npos ||
               device_name.find("q302") != std::string::npos ||
               device_name.find("q350") != std::string::npos) {
      config_cache[device_name] = std::make_unique<Q300ClusterConfig>(
          *static_cast<Q300ClusterConfig *>(config.get()));
    } else if (device_name.find("q250") != std::string::npos ||
               device_name.find("calcium") != std::string::npos) {
      config_cache[device_name] = std::make_unique<CalciumClusterConfig>(
          *static_cast<CalciumClusterConfig *>(config.get()));
    } else {
      config_cache[device_name] = std::make_unique<GpuClusterConfig>(
          *static_cast<GpuClusterConfig *>(config.get()));
    }

    return config;
  }

  std::cerr << "ERROR: GetClusterConfigByName - Failed to load config file: "
            << config_file << std::endl;
  return nullptr;
}

// Helper function to calculate total size from mesh shape
int CalculateMeshSize(const std::vector<int> &mesh_shape) {
  if (mesh_shape.empty()) {
    return 1; // Default size for non-torus devices
  }
  int size = 1;
  for (int dim : mesh_shape) {
    size *= dim;
  }
  return size;
}

// Helper function to validate IntraNodeConfig values
void ValidateIntraNodeConfig(const IntraNodeConfig &config) {
  CHECK_GE(config.size, 1) << "intranode_config.size must be at least 1, got: "
                           << config.size;
  CHECK_GE(config.bandwidth_gbps, 1.0)
      << "intranode_config.bandwidth_gbps must be at least 1.0, got: "
      << config.bandwidth_gbps;
  CHECK_GT(config.efficiency_factor, 0.0)
      << "intranode_config.efficiency_factor must be greater than 0.0, got: "
      << config.efficiency_factor;
  CHECK_LE(config.efficiency_factor, 1.0)
      << "intranode_config.efficiency_factor must be less than or equal to "
         "1.0, got: "
      << config.efficiency_factor;
}

// Forward declarations for helper functions
std::vector<int64_t>
GetDeviceIdsFromOneReplicaGroup(const xla::HloInstruction *instr);
CommType DetermineCommType(const std::vector<int64_t> &device_ids,
                           uint64_t replica_group_size, uint64_t intranode_size,
                           const xla::HloInstruction *instr);
uint64_t GetNumReplicaGroups(const xla::HloInstruction *instr);
uint64_t GetReplicaGroupSize(const xla::HloInstruction *instr);

// Helper function to determine intra-node size and bandwidth based on hardware
// architecture
IntraNodeConfig GetIntraNodeConfigFromDeviceInfo(
    const stream_executor::DeviceDescription &gpu_device_info,
    const std::string &hardware_architecture,
    const std::string &fallback_device_type) {

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

  // Get device configuration using the new system and cast to TpuClusterConfig
  auto device_config = GetClusterConfigByName(device_name, "");
  auto tpu_config = dynamic_cast<TpuClusterConfig *>(device_config.get());
  if (tpu_config) {
    // Calculate intranode_size from intranode_mesh_shape
    int intranode_size = CalculateMeshSize(tpu_config->GetIntranodeMeshShape());
    IntraNodeConfig result = {intranode_size,
                              tpu_config->GetIntranodeBandwidthGbps(),
                              tpu_config->GetIntranodeEfficiencyFactor(),
                              tpu_config->GetReassignBandwidth()};
    ValidateIntraNodeConfig(result);

    return result;
  }

  // Debug: Print available device patterns for troubleshooting
  llvm::outs() << "DEBUG: Hardware architecture '" << device_name
               << "' not found in device configurations.\n";
  llvm::outs() << "DEBUG: Available device patterns: tpuv7e, tpuv7el200, b200, "
                  "b200l200, b300, b300l200, r200, r200l200, r576, r576l200, rcpx, "
                  "rcpxl200, q300, q300l200, q302, q302l200, q350, q350l200\n";
  llvm::outs() << "DEBUG: Make sure config file exists at: configs/"
               << device_name << ".config\n";
  llvm::outs().flush();

  // Fail with error if no exact device configuration match is found
  CHECK(false) << "No exact device configuration match found for hardware "
                  "architecture: '"
               << device_name
               << "'. Please add support for this device type in "
                  "GetIntraNodeConfigFromDeviceInfo function.";
}

int CalculateMaxHop(const std::vector<int64_t> &device_ids,
                    const std::vector<int> &mesh_shape) {
  float max_hop = 0;
  // Go through each dim in the mesh
  for (int i = 0; i < mesh_shape.size(); ++i) {
    max_hop += mesh_shape[i] / 4.0;
  }

  return max_hop;
}

int CalculateAverageHops(const std::vector<int> &mesh_shape) {
  double avg_hops = 0.0;
  for (int dim_size : mesh_shape) {
    avg_hops += dim_size / 4.0;
  }
  return static_cast<int>(std::ceil(avg_hops));
}

// Structure to hold torus hop results
struct TorusHops {
  int copper_hops; // Number of hops over copper links (within pod slice)
  int ici_hops;    // Number of hops over ICI links (between pod slices)
};

// Helper function to convert node ID to 3D coordinates
std::tuple<int, int, int> IdToCoord(int64_t node_id,
                                    const std::vector<int> &dims) {
  int x = node_id % dims[0];
  int y = (node_id / dims[0]) % dims[1];
  int z = (node_id / (dims[0] * dims[1])) % dims[2];
  return std::make_tuple(x, y, z);
}

// Helper function to compute torus distance in 3D
int TorusDistance3D(const std::tuple<int, int, int> &p1,
                    const std::tuple<int, int, int> &p2,
                    const std::vector<int> &dims) {
  int dist = 0;
  int delta_x = std::abs(std::get<0>(p1) - std::get<0>(p2));
  int delta_y = std::abs(std::get<1>(p1) - std::get<1>(p2));
  int delta_z = std::abs(std::get<2>(p1) - std::get<2>(p2));

  dist += std::min(delta_x, dims[0] - delta_x);
  dist += std::min(delta_y, dims[1] - delta_y);
  dist += std::min(delta_z, dims[2] - delta_z);

  return dist;
}

// Calculate hops over copper and ICI links in a 3D torus topology from integer
// node IDs
TorusHops TorusHopsInt(int64_t src_id, int64_t dst_id,
                       const std::vector<int> &pod_dims,
                       const std::vector<int> &slice_dims) {
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
// TpuClusterConfig CalculateCommCost implementation

// Function to calculate TPU communication cost based on 3D torus topology
CommCostStats TpuClusterConfig::CalculateCommCost(
    double per_device_comm_volume,
    const stream_executor::DeviceDescription &tpu_device_info,
    const xla::HloInstruction *instr, uint64_t replica_group_size,
    uint64_t num_replica_groups, const std::vector<int64_t> &device_ids,
    const std::vector<int> &mesh_shape,
    const std::string &hardware_architecture,
    const std::string &fallback_device_type) {
  // Validate inputs
  CHECK_GE(per_device_comm_volume, 0.0)
      << "per_device_comm_volume must be greater than 0.0, got: "
      << per_device_comm_volume;
  CHECK_GE(replica_group_size, 1)
      << "replica_group_size must be at least 1, got: " << replica_group_size;
  CHECK_GE(num_replica_groups, 1)
      << "num_replica_groups must be at least 1, got: " << num_replica_groups;
  CHECK_EQ(mesh_shape.size(), 3) << "Mesh shape must be 3D for TPU torus";
  CHECK_EQ(device_ids.size(), replica_group_size)
      << "Device IDs size must match replica group size";

  // Get device-specific ICI bandwidth from device configuration
  IntraNodeConfig intranode_config = GetIntraNodeConfigFromDeviceInfo(
      tpu_device_info, hardware_architecture, fallback_device_type);

  // Use the current TpuClusterConfig instance's data directly
  int links_per_chip = links_per_chip_;
  std::vector<int> intranode_mesh_shape = intranode_mesh_shape_;
  std::vector<int> internode_mesh_shape = internode_mesh_shape_;
  double intranode_bandwidth_gbps = intranode_bandwidth_gbps_;
  double internode_bandwidth_gbps = internode_bandwidth_gbps_;
  double intranode_efficiency_factor = intranode_efficiency_factor_;
  double internode_efficiency_factor = internode_efficiency_factor_;
  // Calculate sizes from mesh shapes
  int intranode_size = CalculateMeshSize(intranode_mesh_shape);
  int internode_size = CalculateMeshSize(internode_mesh_shape);

  // Calculate bandwidth per link for both intra-node and inter-node (already
  // per-link unidirectional)
  double intranode_bandwidth_per_link_gbps =
      intranode_bandwidth_gbps * intranode_efficiency_factor;
  double internode_bandwidth_per_link_gbps =
      internode_bandwidth_gbps * internode_efficiency_factor;
  // Calculate communication volume per device in GB
  double per_device_comm_vol_gb =
      per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  // Find the bottleneck hop
  // Loop through pair of device ids
  // form pairs to the right. Last node should be paired with the first node.
  double max_comm_cost_us = 0.0;
  TorusHops max_torus_hops = {0, 0};
  int max_hop_src = 0;
  int max_hop_dst = 0;
  double hop_latency_us = 1.0;
  for (size_t i = 0; i < device_ids.size(); ++i) {
    TorusHops torus_hops =
        TorusHopsInt(device_ids[i], device_ids[(i + 1) % device_ids.size()],
                     internode_mesh_shape, intranode_mesh_shape);
    // int number_of_hops = torus_hops.copper_hops + torus_hops.ici_hops;
    double bottleneck_bw = 0.0;
    if (torus_hops.ici_hops > 0) {
      bottleneck_bw = internode_bandwidth_per_link_gbps;
    } else {
      bottleneck_bw = intranode_bandwidth_per_link_gbps;
    }
    double comm_cost_us = ((per_device_comm_vol_gb / bottleneck_bw) * 1e6 + (torus_hops.copper_hops + torus_hops.ici_hops) * hop_latency_us);
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
  if (max_torus_hops.copper_hops > 0) {
    intranode_comm_vol_gb = per_device_comm_vol_gb;
  }
  if (max_torus_hops.ici_hops > 0) {
    internode_comm_vol_gb = per_device_comm_vol_gb;
  }

  double total_comm_vol_gb =
      per_device_comm_vol_gb * max_torus_hops.copper_hops +
      per_device_comm_vol_gb * max_torus_hops.ici_hops;

  CommType comm_type;
  // For TPU, default to ScaleUp for now
  comm_type = CommType::ScaleUp;

  // Debug logging
  if (std::isnan(max_comm_cost_us) || std::isnan(intranode_comm_vol_gb) ||
      std::isnan(total_comm_vol_gb)) {
    llvm::outs()
        << "WARNING: NaN detected in CalculateTpuCommCost for instruction: "
        << instr->ToString() << "\n";
    llvm::outs() << "  per_device_comm_volume: " << per_device_comm_volume
                 << "\n";
    llvm::outs() << "  replica_group_size: " << replica_group_size << "\n";
    llvm::outs() << "  links_per_chip: " << links_per_chip << "\n";
    llvm::outs() << "  copper_hops: " << max_torus_hops.copper_hops << "\n";
    llvm::outs() << "  ici_hops: " << max_torus_hops.ici_hops << "\n";
    llvm::outs() << "  intranode_bandwidth_per_link_gbps: "
                 << intranode_bandwidth_per_link_gbps << "\n";
    llvm::outs() << "  internode_bandwidth_per_link_gbps: "
                 << internode_bandwidth_per_link_gbps << "\n";
    llvm::outs() << "  comm_cost_us: " << max_comm_cost_us << "\n";
    llvm::outs().flush();
  }

  return {
      comm_type,
      max_comm_cost_us,
      intranode_comm_vol_gb,
      internode_comm_vol_gb,
      intranode_bandwidth_per_link_gbps, // intranode_comm_bw_gbps
      internode_bandwidth_per_link_gbps, // internode_comm_bw_gbps
      total_comm_vol_gb,
      per_device_comm_vol_gb, // per_device_comm_vol_gb
      number_of_hops,
      max_comm_cost_us / number_of_hops // per_link_cost_us
  };
}
// GpuClusterConfig CalculateCommCost implementation
CommCostStats GpuClusterConfig::CalculateCommCost(
    double per_device_comm_volume,
    const stream_executor::DeviceDescription &tpu_device_info,
    const xla::HloInstruction *instr, uint64_t replica_group_size,
    uint64_t num_replica_groups, const std::vector<int64_t> &device_ids,
    const std::vector<int> &mesh_shape,
    const std::string &hardware_architecture,
    const std::string &fallback_device_type) {

  // Validate inputs
  if (per_device_comm_volume <= 0.0) {
      CHECK_GE(per_device_comm_volume, 0.0)
          << "per_device_comm_volume must be greater than 0.0, got: "
          << per_device_comm_volume;
    // LOG(FATAL) << "GpuClusterConfig::CalculateCommCost - per_device_comm_volume must be greater than 0.0, got: " << per_device_comm_volume;
  }

  if (device_ids.empty()) {
    LOG(FATAL) << "GpuClusterConfig::CalculateCommCost - device_ids cannot be empty";
  }

  // Convert communication volume to GB
  double per_device_comm_vol_gb =
      per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  // Find the maximum communication cost by checking all device pairs
  double max_comm_cost_us = 0.0;
  int max_hops = 0;
  CommType comm_type = CommType::ScaleUp;
  std::vector<PathComponent> longest_path; // Track the path that incurs the longest communication time

  for (size_t i = 0; i < device_ids.size(); ++i) {
    int src_device_id = static_cast<int>(device_ids[i]);
    int dest_device_id =
        static_cast<int>(device_ids[(i + 1) % device_ids.size()]);

    // Calculate path between devices
    std::vector<PathComponent> path =
        PathBetweenDevices(src_device_id, dest_device_id);
    std::vector<double> bandwidths = PathToBandwidth(path);

    // Calculate communication cost for this path
    double path_cost_us = 0.0;
    if (!bandwidths.empty()) {
      // Find the minimum bandwidth in the path (bottleneck)
      double bottleneck_bw = *std::min_element(bandwidths.begin(), bandwidths.end());

      // Calculate hop count and hop latency
      int hop_count = static_cast<int>(path.size() - 1);
      double hop_latency_us = 1.0;

      // Calculate path cost: per_device_comm_vol_gb / bottleneck_bw + hop_count * hop_latency
      path_cost_us = (per_device_comm_vol_gb / bottleneck_bw) * 1e6 + hop_count * hop_latency_us;
    }

    // Update maximum cost and save the path that incurs the longest communication time
    if (path_cost_us > max_comm_cost_us) {
      max_comm_cost_us = path_cost_us;
      max_hops = static_cast<int>(path.size() - 1);
      longest_path = path; // Save the path that incurs the longest communication time
    }
  }

  // Determine CommType based on the saved communication path
  if (!longest_path.empty()) {
    comm_type = DetermineCommTypeFromPath(longest_path);
  }

  // Calculate communication volumes based on hop counts in each domain
  double intranode_comm_vol_gb = 0.0;
  double internode_comm_vol_gb = 0.0;

  if (!longest_path.empty()) {
    auto hop_counts = CountHopsByDomain(longest_path);
    int scaleup_hops = hop_counts.first;
    int scaleout_hops = hop_counts.second;
    if(scaleout_hops > 0) {
      internode_comm_vol_gb = per_device_comm_vol_gb;
    } else {
      intranode_comm_vol_gb = per_device_comm_vol_gb;
    }
  }

  double total_comm_vol_gb = intranode_comm_vol_gb + internode_comm_vol_gb;

  // Calculate bandwidths based on per-device communication volume and cost
  double avg_bandwidth_gbps = 0.0;
  if (max_comm_cost_us > 0) {
    // Calculate bandwidth as per_device_comm_vol_gb / max_comm_cost_us
    // Convert microseconds to seconds: max_comm_cost_us / 1e6
    avg_bandwidth_gbps = per_device_comm_vol_gb / (max_comm_cost_us / 1e6);
  }

  double per_link_cost_us = (max_hops > 0) ? max_comm_cost_us / max_hops : 0.0;

  return {comm_type,
          max_comm_cost_us,
          intranode_comm_vol_gb,
          internode_comm_vol_gb,
          avg_bandwidth_gbps, // intranode_comm_bw_gbps
          avg_bandwidth_gbps, // internode_comm_bw_gbps
          total_comm_vol_gb,
          per_device_comm_vol_gb, // per_device_comm_vol_gb
          max_hops,
          per_link_cost_us};
}

// ============================================================================
// CalciumClusterConfig Implementation
//
// 4-level fabric: L1 (intra-pod, 8 SoCs) -> L2 (intra-card, 24 SoCs) ->
//                 L3 (intra-server, 192 SoCs) -> L4 (inter-server, RoCE v2).
// ============================================================================

CalciumClusterConfig::CalciumClusterConfig()
    : name_pattern_(""),
      l1_pod_size_(8),
      l1_link_bw_gbytes_(64.0),
      l1_oversubscription_(1.0),
      l1_pods_per_card_(3),
      l2_link_bw_gbytes_(256.0),
      l2_oversubscription_(2.0),
      cards_per_server_(8),
      l3_link_bw_gbytes_(512.0),
      l3_oversubscription_(3.0),
      host_nic_link_bw_gbytes_(800.0),
      l4_per_card_egress_gbytes_(100.0),
      tor_switch_uplink_gbytes_(800.0),
      is_l200_(false),
      optical_port_bw_gbytes_(0.0),
      optical_switch_oversubscription_(1.0),
      socs_per_rack_(0),
      pods_per_rack_(0),
      servers_per_rack_(7),
      num_racks_(1),
      optical_spine_uplink_gbytes_(0.0),
      optical_spine_oversubscription_(1.0),
      intranode_efficiency_factor_(0.95),
      internode_efficiency_factor_(0.85),
      device_id_layout_("row_major_socs_first"),
      hierarchical_allreduce_enabled_(false) {}

std::string CalciumClusterConfig::GetNamePattern() const { return name_pattern_; }

bool CalciumClusterConfig::LoadFromFile(const std::string &config_file_path) {
  std::ifstream file(config_file_path);
  if (!file.is_open()) {
    std::cerr
        << "ERROR: CalciumClusterConfig::LoadFromFile - Cannot open file: "
        << config_file_path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) continue;

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);
    // Trim whitespace.
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    if (key == "name_pattern") {
      name_pattern_ = value;
    } else if (key == "device_id_layout") {
      device_id_layout_ = value;
    } else if (key == "l1_pod_size") {
      l1_pod_size_ = std::stoi(value);
    } else if (key == "l1_link_bw_gbytes") {
      l1_link_bw_gbytes_ = std::stod(value);
    } else if (key == "l1_oversubscription") {
      l1_oversubscription_ = std::stod(value);
    } else if (key == "l1_pods_per_card") {
      l1_pods_per_card_ = std::stoi(value);
    } else if (key == "l2_link_bw_gbytes") {
      l2_link_bw_gbytes_ = std::stod(value);
    } else if (key == "l2_oversubscription") {
      l2_oversubscription_ = std::stod(value);
    } else if (key == "cards_per_server") {
      cards_per_server_ = std::stoi(value);
    } else if (key == "l3_link_bw_gbytes") {
      l3_link_bw_gbytes_ = std::stod(value);
    } else if (key == "l3_oversubscription") {
      l3_oversubscription_ = std::stod(value);
    } else if (key == "host_nic_link_bw_gbytes") {
      host_nic_link_bw_gbytes_ = std::stod(value);
    } else if (key == "l4_per_card_egress_gbytes") {
      l4_per_card_egress_gbytes_ = std::stod(value);
    } else if (key == "tor_switch_uplink_gbytes") {
      tor_switch_uplink_gbytes_ = std::stod(value);
    } else if (key == "intranode_efficiency_factor") {
      intranode_efficiency_factor_ = std::stod(value);
    } else if (key == "internode_efficiency_factor") {
      internode_efficiency_factor_ = std::stod(value);
    } else if (key == "optical_port_bw_gbytes") {
      optical_port_bw_gbytes_ = std::stod(value);
    } else if (key == "optical_switch_oversubscription") {
      optical_switch_oversubscription_ = std::stod(value);
    } else if (key == "socs_per_rack") {
      socs_per_rack_ = std::stoi(value);
    } else if (key == "pods_per_rack") {
      pods_per_rack_ = std::stoi(value);
    } else if (key == "servers_per_rack") {
      servers_per_rack_ = std::stoi(value);
    } else if (key == "num_racks") {
      num_racks_ = std::stoi(value);
    } else if (key == "optical_spine_uplink_gbytes") {
      optical_spine_uplink_gbytes_ = std::stod(value);
    } else if (key == "optical_spine_oversubscription") {
      optical_spine_oversubscription_ = std::stod(value);
    }
    // NOTE: hierarchical AllReduce (Step 9) is intentionally NOT parsed
    // from the .config file. It is controlled exclusively via the FFI
    // parameter `hierarchical_allreduce_enabled` on
    // analytical_latency_calculator_run_with_pipeline (or the matching
    // `--hierarchical-allreduce-enabled` CLI flag). The default in this
    // class is OFF, so legacy callers that don't touch the FFI knob keep
    // the byte-stable flat worst-pair AR cost model.
  }

  file.close();

  if (device_id_layout_ != "row_major_socs_first") {
    std::cerr << "ERROR: CalciumClusterConfig::LoadFromFile - Unsupported "
                 "device_id_layout: "
              << device_id_layout_ << " (only row_major_socs_first supported)"
              << std::endl;
    return false;
  }

  // Detect l200 mode from the optical-port field. q250l200.config sets this
  // to a positive value (e.g. 640); q250.config omits it and leaves it at 0.
  is_l200_ = (optical_port_bw_gbytes_ > 0.0);
  if (is_l200_) {
    if (socs_per_rack_ <= 0 || pods_per_rack_ <= 0) {
      std::cerr << "ERROR: CalciumClusterConfig::LoadFromFile - q250l200 "
                   "config requires socs_per_rack and pods_per_rack > 0."
                << std::endl;
      return false;
    }
    // Multi-rack mode: must supply a non-zero spine uplink so that
    // EffectiveBandwidth on the OpticalSwitch<->OpticalSpine hop is well
    // defined. N == 1 needs no spine (cross-rack branch is unreachable).
    if (num_racks_ > 1 && optical_spine_uplink_gbytes_ <= 0.0) {
      std::cerr << "ERROR: CalciumClusterConfig::LoadFromFile - q250l200 "
                   "config has num_racks="
                << num_racks_
                << " > 1 but optical_spine_uplink_gbytes is unset or <= 0. "
                   "Set optical_spine_uplink_gbytes (e.g. socs_per_rack * "
                   "optical_port_bw_gbytes for non-blocking spine)."
                << std::endl;
      return false;
    }
    if (num_racks_ < 1) num_racks_ = 1;
  }

  return true;
}

CalciumCoord CalciumClusterConfig::DecodeId(int64_t id) const {
  CalciumCoord c{};
  const int64_t socs_per_pod = l1_pod_size_;                          // 8
  const int64_t pods_per_card = l1_pods_per_card_;                    // 3
  const int64_t cards_per_srv = cards_per_server_;                    // 8

  if (is_l200_) {
    // q250l200 layout: id = rack*1344 + pod_in_rack*8 + soc_in_pod.
    // server / card / l1_pod are derived from pod_in_rack so the L4 NIC
    // bundling logic (192 SoCs per server) keeps working unchanged.
    const int64_t socs_per_rk = socs_per_rack_;
    c.rack = static_cast<int>(id / socs_per_rk);
    int64_t in_rack = id % socs_per_rk;
    c.soc_in_pod = static_cast<int>(in_rack % socs_per_pod);
    int64_t pod_in_rack = in_rack / socs_per_pod;
    const int64_t pods_per_server =
        static_cast<int64_t>(pods_per_card) * cards_per_srv;          // 24
    c.server = static_cast<int>(pod_in_rack / pods_per_server);
    int64_t pod_in_server = pod_in_rack % pods_per_server;
    c.card = static_cast<int>(pod_in_server / pods_per_card);
    c.l1_pod = static_cast<int>(pod_in_server % pods_per_card);
    return c;
  }
  // q250 layout: id = server*192 + card*24 + l1_pod*8 + soc_in_pod.
  c.rack = 0;
  c.soc_in_pod = static_cast<int>(id % socs_per_pod);
  id /= socs_per_pod;
  c.l1_pod = static_cast<int>(id % pods_per_card);
  id /= pods_per_card;
  c.card = static_cast<int>(id % cards_per_srv);
  id /= cards_per_srv;
  c.server = static_cast<int>(id);
  return c;
}

std::vector<PathComponent>
CalciumClusterConfig::PathBetweenDevices(int64_t src, int64_t dst) const {
  CalciumCoord s = DecodeId(src);
  CalciumCoord d = DecodeId(dst);

  // Same SoC (works for both q250 and q250l200; rack=0 always for q250).
  if (s.rack == d.rack && s.server == d.server && s.card == d.card &&
      s.l1_pod == d.l1_pod && s.soc_in_pod == d.soc_in_pod) {
    return {PathComponent::GPU};
  }

  if (is_l200_) {
    // q250l200: optical interposer (8-SoC pod) + rack-wide optical switch.
    // SoCs are PathComponent::GPU here (the enum is shared across arches);
    // see Calcium docs for the SoC naming.
    //
    // Intra-pod: 1 hop direct SoC<->SoC on the optical interposer.
    if (s.rack == d.rack && s.server == d.server && s.card == d.card &&
        s.l1_pod == d.l1_pod) {
      return {PathComponent::GPU, PathComponent::GPU};
    }
    // Cross-pod intra-rack: 2 hops via the rack-wide optical switch.
    if (s.rack == d.rack) {
      return {PathComponent::GPU, PathComponent::OpticalSwitch,
              PathComponent::GPU};
    }
    // Cross-rack: stay in the scale-up optical domain via the rack-aggregation
    // OpticalSpine. 4 hops total, no L4 RoCE / ToR. Reaching this branch
    // requires num_racks_ > 1 (otherwise every legal id has rack == 0 and
    // s.rack == d.rack always holds, preserving the N=1 byte-stable behavior).
    return {PathComponent::GPU, PathComponent::OpticalSwitch,
            PathComponent::OpticalSpine, PathComponent::OpticalSwitch,
            PathComponent::GPU};
  }

  // q250 (legacy 4-level PCIe + RoCE).
  // Intra-pod: 2 hops.
  if (s.server == d.server && s.card == d.card && s.l1_pod == d.l1_pod) {
    return {PathComponent::GPU, PathComponent::L1Switch, PathComponent::GPU};
  }
  // Intra-card, cross-pod: 4 hops.
  if (s.server == d.server && s.card == d.card) {
    return {PathComponent::GPU, PathComponent::L1Switch,
            PathComponent::L2Switch, PathComponent::L1Switch,
            PathComponent::GPU};
  }
  // Intra-server, cross-card: 6 hops.
  if (s.server == d.server) {
    return {PathComponent::GPU, PathComponent::L1Switch,
            PathComponent::L2Switch, PathComponent::L3Switch,
            PathComponent::L2Switch, PathComponent::L1Switch,
            PathComponent::GPU};
  }
  // Inter-server: 10 hops via ToR.
  return {PathComponent::GPU,      PathComponent::L1Switch,
          PathComponent::L2Switch, PathComponent::L3Switch,
          PathComponent::NIC,      PathComponent::ToRSwitch,
          PathComponent::NIC,      PathComponent::L3Switch,
          PathComponent::L2Switch, PathComponent::L1Switch,
          PathComponent::GPU};
}

namespace {
// Local helper: order-insensitive link match.
bool IsLink(PathComponent a, PathComponent b, PathComponent x, PathComponent y) {
  return (a == x && b == y) || (a == y && b == x);
}
}  // namespace

double CalciumClusterConfig::StaticLinkBandwidth(PathComponent a,
                                                 PathComponent b) const {
  // q250l200 optical fabric edges. Both intra-pod direct (SoC<->SoC on the
  // interposer) and SoC<->switch are bounded by the SoC's single optical
  // port BW.
  if (IsLink(a, b, PathComponent::GPU, PathComponent::GPU))
    return optical_port_bw_gbytes_;
  if (IsLink(a, b, PathComponent::GPU, PathComponent::OpticalSwitch))
    return optical_port_bw_gbytes_;
  // q250l200 cross-rack optical spine: the rack-wide OpticalSwitch uplinks
  // into a datacenter-scale OpticalSpine. The link_bw is the per-rack uplink
  // aggregate; per-SoC effective BW is link_bw / devices_sharing (capped at
  // the SoC's own optical_port_bw_gbytes_ leaf cap in EffectiveBandwidth).
  if (IsLink(a, b, PathComponent::OpticalSwitch, PathComponent::OpticalSpine))
    return optical_spine_uplink_gbytes_;
  // q250l200 cross-rack: SoC connects directly to its server's L4 NIC
  // (per-server bundle, 800 GB/s aggregate). Leaf cap (the SoC's egress
  // port) is enforced separately in EffectiveBandwidth. This path is no
  // longer emitted by PathBetweenDevices but the entry is retained so that
  // direct callers (diagnostics, tests) keep working.
  if (IsLink(a, b, PathComponent::GPU, PathComponent::NIC))
    return host_nic_link_bw_gbytes_;
  // q250 4-level PCIe + L4 RoCE.
  if (IsLink(a, b, PathComponent::GPU, PathComponent::L1Switch))
    return l1_link_bw_gbytes_;
  if (IsLink(a, b, PathComponent::L1Switch, PathComponent::L2Switch))
    return l2_link_bw_gbytes_;
  if (IsLink(a, b, PathComponent::L2Switch, PathComponent::L3Switch))
    return l3_link_bw_gbytes_;
  if (IsLink(a, b, PathComponent::L3Switch, PathComponent::NIC))
    return host_nic_link_bw_gbytes_;
  if (IsLink(a, b, PathComponent::NIC, PathComponent::ToRSwitch))
    return tor_switch_uplink_gbytes_;
  // Defensive fallback for unexpected adjacent pairs in well-formed Calcium
  // paths. Returning 1.0 makes path_cost = volume_gb / 1.0 -> seconds, which
  // is large enough to surface clearly as a bug if ever hit.
  return 1.0;
}

double CalciumClusterConfig::StaticOversubscription(PathComponent a,
                                                    PathComponent b) const {
  if (IsLink(a, b, PathComponent::L1Switch, PathComponent::L2Switch))
    return l2_oversubscription_;
  if (IsLink(a, b, PathComponent::L2Switch, PathComponent::L3Switch))
    return l3_oversubscription_;
  if (IsLink(a, b, PathComponent::GPU, PathComponent::OpticalSwitch))
    return optical_switch_oversubscription_;
  if (IsLink(a, b, PathComponent::OpticalSwitch, PathComponent::OpticalSpine))
    return optical_spine_oversubscription_;
  // SoC<->SoC interposer link (q250l200), L1 (non-blocking), host-internal,
  // L4 ToR uplinks, SoC<->NIC: no internal oversub.
  return 1.0;
}

double CalciumClusterConfig::EffectiveBandwidth(PathComponent a,
                                                PathComponent b,
                                                int devices_sharing) const {
  if (devices_sharing < 1) devices_sharing = 1;
  // Per-SoC effective bandwidth at this hop is min(leaf-link capacity,
  // shared upstream / devices sharing). The structural oversubscription
  // recorded in the .config (StaticOversubscription) is already implied by
  // (link_bw, sharing) at full-subtree saturation - dividing by it here would
  // double-count. The leaf-link cap prevents single-SoC and small replica
  // groups from being credited with more upstream BW than the SoC's own
  // dedicated egress port can sustain.
  //
  // Leaf cap differs by fabric:
  //   q250    -> l1_link_bw_gbytes_     (PCIe Gen5 x16 SoC<->L1 = 64 GB/s)
  //   q250l200 -> optical_port_bw_gbytes_ (optical port, e.g. 640 GB/s)
  const double link_bw = StaticLinkBandwidth(a, b);
  const double leaf_bw = is_l200_ ? optical_port_bw_gbytes_
                                  : l1_link_bw_gbytes_;
  const double per_soc_share = link_bw / static_cast<double>(devices_sharing);
  return std::min(leaf_bw, per_soc_share);
}

int CalciumClusterConfig::ComputeDevicesSharingHop(
    PathComponent a, PathComponent b,
    const std::vector<int64_t> &device_ids) const {
  // Per-SoC dedicated edges, never shared (PathComponent::GPU is the SoC):
  //   q250    : SoC<->L1Switch
  //   q250l200: SoC<->SoC (interposer-direct), SoC<->OpticalSwitch
  if (IsLink(a, b, PathComponent::GPU, PathComponent::L1Switch)) return 1;
  if (IsLink(a, b, PathComponent::GPU, PathComponent::GPU)) return 1;
  if (IsLink(a, b, PathComponent::GPU, PathComponent::OpticalSwitch)) return 1;

  auto MaxGroupSize = [&](auto key_fn) -> int {
    std::unordered_map<int64_t, int> counts;
    for (int64_t id : device_ids) {
      counts[key_fn(DecodeId(id))]++;
    }
    int max_n = 0;
    for (const auto &kv : counts) {
      if (kv.second > max_n) max_n = kv.second;
    }
    return std::max(1, max_n);
  };

  // L1<->L2: per-pod L2 uplink, shared by all 8 SoCs in the pod that
  // participate in this collective.
  if (IsLink(a, b, PathComponent::L1Switch, PathComponent::L2Switch)) {
    return MaxGroupSize([this](CalciumCoord c) -> int64_t {
      return (static_cast<int64_t>(c.server) * cards_per_server_ + c.card) *
                 l1_pods_per_card_ +
             c.l1_pod;
    });
  }
  // L2<->L3: per-card L3 uplink, shared by all 24 SoCs on the card that
  // participate in this collective.
  if (IsLink(a, b, PathComponent::L2Switch, PathComponent::L3Switch)) {
    return MaxGroupSize([this](CalciumCoord c) -> int64_t {
      return static_cast<int64_t>(c.server) * cards_per_server_ + c.card;
    });
  }
  // L3<->NIC, NIC<->ToR (q250) and SoC<->NIC (q250l200): per-server egress,
  // shared by all 192 SoCs on the server that participate in this
  // collective. Group by (rack, server) so q250 (rack==0 always) collapses
  // to per-server and q250l200 separates rack-by-rack.
  if (IsLink(a, b, PathComponent::L3Switch, PathComponent::NIC) ||
      IsLink(a, b, PathComponent::NIC, PathComponent::ToRSwitch) ||
      IsLink(a, b, PathComponent::GPU, PathComponent::NIC)) {
    return MaxGroupSize([this](CalciumCoord c) -> int64_t {
      return static_cast<int64_t>(c.rack) * servers_per_rack_ + c.server;
    });
  }
  // OpticalSwitch<->OpticalSpine (q250l200 cross-rack): per-rack uplink to
  // the datacenter-scale spine, shared by all SoCs in the rack that
  // participate in this collective.
  if (IsLink(a, b, PathComponent::OpticalSwitch, PathComponent::OpticalSpine)) {
    return MaxGroupSize([](CalciumCoord c) -> int64_t {
      return c.rack;
    });
  }
  return 1;
}

double CalciumClusterConfig::CalculatePipelineHandoffCost(
    int64_t activation_bytes, int devices_per_stage) const {
  // Backward-compatible wrapper: single boundary, no overlap. Equivalent to
  // the per-boundary computation for k=0 only. For S=2 this is exact; for
  // S>2 callers should switch to CalculatePipelineHandoffCosts(...) so the
  // slowest boundary drives steady-state cadence.
  PipelineHandoffStats s = CalculatePipelineHandoffCosts(
      activation_bytes, devices_per_stage,
      /*num_pipeline_stages=*/2,
      /*pipeline_comm_overlap_factor=*/0.0);
  return s.max_us;
}

PipelineHandoffStats CalciumClusterConfig::CalculatePipelineHandoffCosts(
    int64_t activation_bytes, int devices_per_stage,
    int num_pipeline_stages, double pipeline_comm_overlap_factor) const {
  PipelineHandoffStats out;
  out.pipeline_comm_overlap_factor =
      std::min(1.0, std::max(0.0, pipeline_comm_overlap_factor));
  if (devices_per_stage <= 0 || activation_bytes <= 0 ||
      num_pipeline_stages < 2) {
    return out;
  }
  const int boundaries = num_pipeline_stages - 1;
  out.num_boundaries = boundaries;
  out.per_boundary_us.reserve(boundaries);
  out.per_boundary_raw_us.reserve(boundaries);

  for (int k = 0; k < boundaries; ++k) {
    const int64_t src = static_cast<int64_t>(k) * devices_per_stage;
    const int64_t dst =
        static_cast<int64_t>(k + 1) * devices_per_stage;
    const std::vector<PathComponent> path = PathBetweenDevices(src, dst);
    if (path.size() < 2) {
      out.per_boundary_us.push_back(0.0);
      out.per_boundary_raw_us.push_back(0.0);
      continue;
    }
    double bottleneck_gbps = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      // P2P (1-to-1) handoff: sharing=1 on every hop.
      const double bw =
          EffectiveBandwidth(path[i], path[i + 1], /*sharing=*/1);
      if (bw < bottleneck_gbps) bottleneck_gbps = bw;
    }
    if (!std::isfinite(bottleneck_gbps) || bottleneck_gbps <= 0.0) {
      out.per_boundary_us.push_back(0.0);
      out.per_boundary_raw_us.push_back(0.0);
      continue;
    }
    const double eff_bw = bottleneck_gbps * internode_efficiency_factor_;
    const double bytes_us =
        (static_cast<double>(activation_bytes) / 1e9) / eff_bw * 1e6;
    const double hops_us =
        static_cast<double>(path.size() - 1) * 1.0;  // 1 us per hop.
    const double t_raw = bytes_us + hops_us;
    const double t_eff = t_raw * (1.0 - out.pipeline_comm_overlap_factor);
    out.per_boundary_us.push_back(t_eff);
    out.per_boundary_raw_us.push_back(t_raw);
  }

  for (double t : out.per_boundary_us) {
    out.sum_us += t;
    if (t > out.max_us) out.max_us = t;
  }
  for (double t : out.per_boundary_raw_us) {
    out.raw_sum_us += t;
    if (t > out.raw_max_us) out.raw_max_us = t;
  }
  out.avg_us = (boundaries > 0) ? (out.sum_us / boundaries) : 0.0;
  return out;
}

CommType CalciumClusterConfig::DetermineCommTypeFromPath(
    const std::vector<PathComponent> &path) const {
  bool has_tor = false;
  bool has_l3 = false;
  bool has_optical = false;
  for (PathComponent c : path) {
    if (c == PathComponent::ToRSwitch) has_tor = true;
    if (c == PathComponent::L3Switch) has_l3 = true;
    if (c == PathComponent::OpticalSwitch) has_optical = true;
  }
  // Cross-rack (L4 via ToR) is scale-out for both q250 and q250l200; the
  // entire intra-rack/intra-server fabric is treated as scale-up.
  if (has_tor) return CommType::ScaleOut;
  // q250l200 intra-rack via the optical switch is scale-up by design.
  // q250  intra-server via L3 is also scale-up. Both share the ScaleUp
  // category in the existing CommType vocabulary.
  (void)has_l3;
  (void)has_optical;
  return CommType::ScaleUp;
}

// =====================================================================
// Step 9: hierarchical AllReduce helpers.
//
// All four are no-ops when hierarchical_allreduce_enabled_ is false; that
// is enforced by the caller (CalculateCommCost). The helpers themselves
// don't read the flag - they're pure cost functions, so they can be called
// from tests/smokes regardless.
// =====================================================================

namespace {

// Identifies "leaf" hops: per-SoC dedicated egress edges. These are NOT
// shared across simultaneous parallel rails - each rail uses a different
// SoC's port - so the parallel_rails multiplier MUST NOT inflate sharing
// here. Other links (L1<->L2, L2<->L3, L3<->NIC, NIC<->ToR, etc.) are
// shared by multiple SoCs in their fan-in subtree and DO see parallel_rails
// contention.
bool IsLeafHopForRails(PathComponent a, PathComponent b) {
  auto eq = [](PathComponent x, PathComponent y, PathComponent u,
               PathComponent v) { return (x == u && y == v) ||
                                         (x == v && y == u); };
  // q250: SoC <-> L1Switch is the leaf.
  if (eq(a, b, PathComponent::GPU, PathComponent::L1Switch)) return true;
  // q250l200: SoC <-> SoC interposer (intra-pod), SoC <-> OpticalSwitch
  // (cross-pod intra-rack), SoC <-> NIC (cross-rack). All three are
  // per-SoC dedicated optical / NIC ports.
  if (eq(a, b, PathComponent::GPU, PathComponent::GPU)) return true;
  if (eq(a, b, PathComponent::GPU, PathComponent::OpticalSwitch)) return true;
  if (eq(a, b, PathComponent::GPU, PathComponent::NIC)) return true;
  return false;
}

}  // namespace

std::vector<std::vector<int64_t>>
CalciumClusterConfig::PartitionAtTier(
    const std::vector<int64_t>& device_ids, Tier tier) const {
  // Map each device id to its tier-key, preserving id order within a key
  // so the first-encountered id of a subgroup remains its representative.
  std::unordered_map<int64_t, std::vector<int64_t>> by_key;
  std::vector<int64_t> key_order;
  by_key.reserve(device_ids.size());
  for (int64_t id : device_ids) {
    CalciumCoord c = DecodeId(id);
    int64_t key = 0;
    switch (tier) {
      case Tier::Pod:
        // (rack, server, card, l1_pod)
        key = (((static_cast<int64_t>(c.rack) * servers_per_rack_) + c.server)
                  * cards_per_server_ + c.card) * l1_pods_per_card_ + c.l1_pod;
        break;
      case Tier::Card:
        // (rack, server, card)
        key = ((static_cast<int64_t>(c.rack) * servers_per_rack_) + c.server)
                  * cards_per_server_ + c.card;
        break;
      case Tier::Server:
        // (rack, server)
        key = static_cast<int64_t>(c.rack) * servers_per_rack_ + c.server;
        break;
      case Tier::Rack:
        key = c.rack;
        break;
    }
    auto it = by_key.find(key);
    if (it == by_key.end()) {
      by_key.emplace(key, std::vector<int64_t>{id});
      key_order.push_back(key);
    } else {
      it->second.push_back(id);
    }
  }
  std::vector<std::vector<int64_t>> out;
  out.reserve(key_order.size());
  for (int64_t k : key_order) out.push_back(std::move(by_key[k]));
  return out;
}

std::vector<CalciumClusterConfig::Tier>
CalciumClusterConfig::AutoSelectTiers(
    const std::vector<int64_t>& device_ids) const {
  if (device_ids.size() <= 1) return {};
  bool multi_pod = false;
  bool multi_card = false;
  bool multi_server = false;
  bool multi_rack = false;
  CalciumCoord first = DecodeId(device_ids.front());
  for (int64_t id : device_ids) {
    CalciumCoord c = DecodeId(id);
    if (c.rack != first.rack) multi_rack = true;
    if (c.rack != first.rack || c.server != first.server) multi_server = true;
    if (c.rack != first.rack || c.server != first.server ||
        c.card != first.card) multi_card = true;
    if (c.rack != first.rack || c.server != first.server ||
        c.card != first.card || c.l1_pod != first.l1_pod) multi_pod = true;
  }
  // Whole group fits in a single pod -> nothing to gain from decomposition;
  // flat AR is already correct (single fast tier).
  if (!multi_pod) return {};

  // q250l200 hybrid optical: pod boundary is the only fast/slow split
  // worth exploiting on intra-rack groups; multi-rack adds the L4 NIC tier.
  if (is_l200_) {
    if (multi_rack) return {Tier::Pod, Tier::Rack};
    return {Tier::Pod};
  }
  // q250 4-level PCIe + L4 RoCE.
  if (multi_server) return {Tier::Pod, Tier::Card, Tier::Server};
  if (multi_card) return {Tier::Pod, Tier::Card};
  return {Tier::Pod};
}

double CalciumClusterConfig::SubgroupRsAgCostUs(
    const std::vector<int64_t>& subgroup, double per_device_bytes,
    int parallel_rails) const {
  if (subgroup.size() <= 1 || per_device_bytes <= 0.0) return 0.0;
  if (parallel_rails < 1) parallel_rails = 1;

  // Worst-pair path inside the subgroup. For a homogeneous Calcium tier
  // (pod/card/server/rack) all pairs share the same path topology, so any
  // pair would do; we still iterate to match flat-AR worst-pair semantics
  // and to be robust against degenerate subgroups.
  double max_us = 0.0;
  for (size_t i = 0; i < subgroup.size(); ++i) {
    const int64_t src = subgroup[i];
    const int64_t dst = subgroup[(i + 1) % subgroup.size()];
    std::vector<PathComponent> path = PathBetweenDevices(src, dst);
    if (path.size() < 2) continue;

    double bottleneck_gbps = std::numeric_limits<double>::infinity();
    for (size_t h = 0; h + 1 < path.size(); ++h) {
      const int natural =
          ComputeDevicesSharingHop(path[h], path[h + 1], subgroup);
      // Only inflate sharing on shared upstream hops. Leaf hops are per-SoC
      // dedicated and parallel rails use different SoC ports - sharing on
      // that hop is always 1 regardless of parallel_rails.
      const int effective =
          IsLeafHopForRails(path[h], path[h + 1])
              ? natural
              : natural * parallel_rails;
      const double bw = EffectiveBandwidth(path[h], path[h + 1], effective);
      if (bw < bottleneck_gbps) bottleneck_gbps = bw;
    }
    if (!std::isfinite(bottleneck_gbps) || bottleneck_gbps <= 0.0) continue;

    const int hop_count = static_cast<int>(path.size() - 1);
    const double bytes_us =
        (per_device_bytes / (1024.0 * 1024.0 * 1024.0)) /
        bottleneck_gbps * 1e6;
    const double t_us = bytes_us + hop_count * 1.0;  // 1 us / hop
    if (t_us > max_us) max_us = t_us;
  }
  return max_us;
}

double CalciumClusterConfig::FlatAllReduceCostUs(
    const std::vector<int64_t>& device_ids, double per_device_bytes,
    int parallel_rails) const {
  // Identical structure to the legacy CalculateCommCost worst-pair loop,
  // factored out so the recursion's base case can reuse it. Top-level
  // callers pass parallel_rails=1, which is byte-stable vs. the legacy
  // path. We charge the FULL per-device AR volume at the worst-pair
  // bottleneck (this matches the legacy semantics; the volume itself
  // already encodes 2*(N-1)/N from the caller).
  if (device_ids.size() <= 1 || per_device_bytes <= 0.0) return 0.0;
  if (parallel_rails < 1) parallel_rails = 1;

  double max_us = 0.0;
  for (size_t i = 0; i < device_ids.size(); ++i) {
    const int64_t src = device_ids[i];
    const int64_t dst = device_ids[(i + 1) % device_ids.size()];
    std::vector<PathComponent> path = PathBetweenDevices(src, dst);
    if (path.size() < 2) continue;

    double bottleneck_gbps = std::numeric_limits<double>::infinity();
    for (size_t h = 0; h + 1 < path.size(); ++h) {
      const int natural =
          ComputeDevicesSharingHop(path[h], path[h + 1], device_ids);
      const int effective =
          IsLeafHopForRails(path[h], path[h + 1])
              ? natural
              : natural * parallel_rails;
      const double bw = EffectiveBandwidth(path[h], path[h + 1], effective);
      if (bw < bottleneck_gbps) bottleneck_gbps = bw;
    }
    if (!std::isfinite(bottleneck_gbps) || bottleneck_gbps <= 0.0) continue;

    const int hop_count = static_cast<int>(path.size() - 1);
    const double bytes_us =
        (per_device_bytes / (1024.0 * 1024.0 * 1024.0)) /
        bottleneck_gbps * 1e6;
    const double t_us = bytes_us + hop_count * 1.0;
    if (t_us > max_us) max_us = t_us;
  }
  return max_us;
}

double CalciumClusterConfig::HierarchicalAllReduceCostUs(
    const std::vector<int64_t>& device_ids, double tensor_bytes,
    const std::vector<Tier>& tiers, int parallel_rails) const {
  // NB: this function takes the TENSOR size T (not the per-device AR
  // volume V = 2(N-1)/N * T). The plan's stage-volume formulas all
  // reference T directly, and the recursive cross-subgroup AR runs on
  // the (T/k)-shard tensor. The single conversion happens at the top-
  // level call site in CalculateCommCost.
  if (device_ids.size() <= 1 || tensor_bytes <= 0.0) return 0.0;
  if (parallel_rails < 1) parallel_rails = 1;

  // Base case: flat AR over the (now-homogeneous) group on the current
  // tensor. Convert tensor -> per-device AR volume (2(N-1)/N * T) so
  // FlatAllReduceCostUs can charge it like a normal flat AR.
  if (tiers.empty()) {
    const double N = static_cast<double>(device_ids.size());
    const double per_dev_vol = 2.0 * (N - 1.0) / N * tensor_bytes;
    return FlatAllReduceCostUs(device_ids, per_dev_vol, parallel_rails);
  }

  std::vector<std::vector<int64_t>> subgroups =
      PartitionAtTier(device_ids, tiers.front());
  std::vector<Tier> rest(tiers.begin() + 1, tiers.end());

  // Whole group fits in one subgroup at this tier: descend without paying
  // the decomposition.
  if (subgroups.size() <= 1) {
    return HierarchicalAllReduceCostUs(device_ids, tensor_bytes, rest,
                                       parallel_rails);
  }

  // Stage 1 - intra-subgroup ReduceScatter. All subgroups run in parallel;
  // the cost is the slowest subgroup (worst-pair-style charging).
  // Per-SoC RS volume = T * (k-1)/k where k = subgroup size.
  double t_rs = 0.0;
  for (const auto& sg : subgroups) {
    if (sg.size() <= 1) continue;
    const double k = static_cast<double>(sg.size());
    const double rs_vol = tensor_bytes * (k - 1.0) / k;
    t_rs = std::max(t_rs, SubgroupRsAgCostUs(sg, rs_vol, parallel_rails));
  }

  // Stage 2 - cross-subgroup AR on the (T/k)-shard. Use the SMALLEST
  // subgroup size as `k` (most pessimistic shard size for unequal
  // subgroups; equals the common k for equal subgroups). The recursive
  // call gets parallel_rails *= k because the outer subgroup runs k
  // simultaneous shard-rails over the up-tier links.
  int k_min = static_cast<int>(subgroups.front().size());
  for (const auto& sg : subgroups) {
    int s = static_cast<int>(sg.size());
    if (s > 0 && s < k_min) k_min = s;
  }
  if (k_min < 1) k_min = 1;
  std::vector<int64_t> reps;
  reps.reserve(subgroups.size());
  for (const auto& sg : subgroups) reps.push_back(sg.front());
  const double t_cross = HierarchicalAllReduceCostUs(
      reps, tensor_bytes / static_cast<double>(k_min), rest,
      parallel_rails * k_min);

  // Stage 3 - intra-subgroup AllGather. Symmetric to Stage 1.
  const double t_ag = t_rs;

  return t_rs + t_cross + t_ag;
}

CommCostStats CalciumClusterConfig::CalculateCommCost(
    double per_device_comm_volume,
    const stream_executor::DeviceDescription & /*device_info*/,
    const xla::HloInstruction *instr, uint64_t replica_group_size,
    uint64_t /*num_replica_groups*/, const std::vector<int64_t> &device_ids,
    const std::vector<int> & /*mesh_shape*/,
    const std::string & /*hardware_architecture*/,
    const std::string & /*fallback_device_type*/) {
  if (per_device_comm_volume < 0.0) {
    CHECK_GE(per_device_comm_volume, 0.0)
        << "per_device_comm_volume must be >= 0, got: "
        << per_device_comm_volume;
  }
  if (device_ids.empty()) {
    LOG(FATAL) << "CalciumClusterConfig::CalculateCommCost - device_ids "
                  "cannot be empty";
  }

  const double per_device_comm_vol_gb =
      per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  // Find the worst-case pair in the replica group: largest hop count, and
  // among those, smallest effective bandwidth on the bottleneck hop.
  double max_comm_cost_us = 0.0;
  int max_hops = 0;
  CommType comm_type = CommType::ScaleUp;
  std::vector<PathComponent> longest_path;
  double longest_bottleneck_gbps = 0.0;

  for (size_t i = 0; i < device_ids.size(); ++i) {
    const int64_t src = device_ids[i];
    const int64_t dst = device_ids[(i + 1) % device_ids.size()];
    std::vector<PathComponent> path = PathBetweenDevices(src, dst);
    if (path.size() < 2) continue;

    // Per-hop effective bandwidth with replica-group contention.
    double bottleneck_gbps = std::numeric_limits<double>::infinity();
    for (size_t h = 0; h + 1 < path.size(); ++h) {
      const int sharing =
          ComputeDevicesSharingHop(path[h], path[h + 1], device_ids);
      const double bw = EffectiveBandwidth(path[h], path[h + 1], sharing);
      if (bw < bottleneck_gbps) bottleneck_gbps = bw;
    }
    if (!std::isfinite(bottleneck_gbps) || bottleneck_gbps <= 0.0) continue;

    const int hop_count = static_cast<int>(path.size() - 1);
    // hop latency 1us per hop, matching GpuClusterConfig.
    const double hop_latency_us = 1.0;
    const double path_cost_us =
        (per_device_comm_vol_gb / bottleneck_gbps) * 1e6 +
        hop_count * hop_latency_us;

    if (path_cost_us > max_comm_cost_us) {
      max_comm_cost_us = path_cost_us;
      max_hops = hop_count;
      longest_path = path;
      longest_bottleneck_gbps = bottleneck_gbps;
    }
  }

  if (!longest_path.empty()) {
    comm_type = DetermineCommTypeFromPath(longest_path);
  }

  // Step 9: hierarchical AllReduce. Replaces `max_comm_cost_us` only when
  //   (a) the per-config flag `hierarchical_allreduce_enabled=1` is set, AND
  //   (b) the opcode is kAllReduce / kAllReduceStart, AND
  //   (c) AutoSelectTiers picks at least one tier (i.e. the replica group
  //       actually crosses a tier boundary - single-pod groups stay flat).
  // Every other code path (other opcodes, other arches, flag off) is
  // byte-stable. `comm_type` and `max_hops` are still derived from the flat
  // worst-pair `longest_path` to preserve CSV classification semantics
  // (plan §4.6).
  if (hierarchical_allreduce_enabled_ && instr != nullptr &&
      (instr->opcode() == xla::HloOpcode::kAllReduce ||
       instr->opcode() == xla::HloOpcode::kAllReduceStart)) {
    std::vector<Tier> tiers = AutoSelectTiers(device_ids);
    if (!tiers.empty()) {
      // Recover tensor bytes T from the per-device AR volume V:
      //   V = 2(N-1)/N * T   =>   T = V * N / (2(N-1))
      // The hierarchical recursion expresses stage volumes in T directly;
      // see HierarchicalAllReduceCostUs for the rationale.
      const double N = static_cast<double>(device_ids.size());
      const double tensor_bytes =
          (N > 1.0)
              ? per_device_comm_volume * N / (2.0 * (N - 1.0))
              : per_device_comm_volume;
      const double hier_us = HierarchicalAllReduceCostUs(
          device_ids, tensor_bytes, tiers, /*parallel_rails=*/1);
      // Monotonicity invariant (plan §5.3 #2): hierarchical schedule must
      // never cost more than the flat worst-pair schedule for the same
      // group/volume. If it ever does we have a modeling bug; clamp and
      // warn rather than crash so production callers don't suddenly fail.
      if (hier_us > max_comm_cost_us + 1e-6 && max_comm_cost_us > 0.0) {
        LOG(WARNING) << "CalciumClusterConfig: hierarchical AR ("
                     << hier_us << " us) > flat AR (" << max_comm_cost_us
                     << " us); falling back to flat. group_size="
                     << device_ids.size()
                     << " per_device_volume=" << per_device_comm_volume;
      } else if (hier_us > 0.0) {
        max_comm_cost_us = hier_us;
      }
    }
  }

  // Apply efficiency factor based on intra/inter-node classification.
  // Note: per_device_comm_volume already incorporates the 2*(N-1)/N collective
  // factor from the caller; we must NOT re-apply it here.
  const double efficiency = (comm_type == CommType::ScaleOut)
                                ? internode_efficiency_factor_
                                : intranode_efficiency_factor_;
  if (efficiency > 0.0 && efficiency < 1.0 && max_comm_cost_us > 0.0) {
    max_comm_cost_us /= efficiency;
  }
  (void)replica_group_size;  // Reserved for future use (e.g., per-protocol tuning).

  double intranode_comm_vol_gb = 0.0;
  double internode_comm_vol_gb = 0.0;
  if (comm_type == CommType::ScaleOut) {
    internode_comm_vol_gb = per_device_comm_vol_gb;
  } else {
    intranode_comm_vol_gb = per_device_comm_vol_gb;
  }
  const double total_comm_vol_gb =
      intranode_comm_vol_gb + internode_comm_vol_gb;

  const double avg_bandwidth_gbps =
      (max_comm_cost_us > 0.0)
          ? per_device_comm_vol_gb / (max_comm_cost_us / 1e6)
          : 0.0;
  const double per_link_cost_us =
      (max_hops > 0) ? max_comm_cost_us / max_hops : 0.0;
  (void)longest_bottleneck_gbps;  // available for future diagnostics

  return {comm_type,
          max_comm_cost_us,
          intranode_comm_vol_gb,
          internode_comm_vol_gb,
          avg_bandwidth_gbps,  // intranode_comm_bw_gbps
          avg_bandwidth_gbps,  // internode_comm_bw_gbps
          total_comm_vol_gb,
          per_device_comm_vol_gb,
          max_hops,
          per_link_cost_us};
}

// ============================================================================
// Q300ClusterConfig Implementation
//
// Flat 1-tier ESUN Ethernet scale-up: any intra-rack pair is
// SoC -> EthSwitch -> SoC (2 hops). Cross-rack is rejected at load.
// ============================================================================

Q300ClusterConfig::Q300ClusterConfig()
    : name_pattern_(""),
      socs_per_card_(8),
      cards_per_rack_(72),
      socs_per_rack_(576),
      num_racks_(1),
      eic_port_bw_gbytes_(224.0),
      fabric_oversubscription_(1.0),
      intranode_efficiency_factor_(0.95),
      internode_efficiency_factor_(1.0),
      device_id_layout_("row_major_socs_first") {}

std::string Q300ClusterConfig::GetNamePattern() const { return name_pattern_; }

bool Q300ClusterConfig::LoadFromFile(const std::string &config_file_path) {
  std::ifstream file(config_file_path);
  if (!file.is_open()) {
    std::cerr << "ERROR: Q300ClusterConfig::LoadFromFile - Cannot open file: "
              << config_file_path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t eq_pos = line.find('=');
    if (eq_pos == std::string::npos) continue;

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    if (key == "name_pattern") {
      name_pattern_ = value;
    } else if (key == "device_id_layout") {
      device_id_layout_ = value;
    } else if (key == "socs_per_card") {
      socs_per_card_ = std::stoi(value);
    } else if (key == "cards_per_rack") {
      cards_per_rack_ = std::stoi(value);
    } else if (key == "socs_per_rack") {
      socs_per_rack_ = std::stoi(value);
    } else if (key == "num_racks") {
      num_racks_ = std::stoi(value);
    } else if (key == "eic_port_bw_gbytes") {
      eic_port_bw_gbytes_ = std::stod(value);
    } else if (key == "fabric_oversubscription") {
      fabric_oversubscription_ = std::stod(value);
    } else if (key == "parallel_rails") {
      // Accepted for backward-compat with older configs; intentionally
      // discarded. The 8/16 fabric mesh planes are not a per-SoC BW
      // multiplier (per-SoC egress is already inside eic_port_bw_gbytes).
      (void)value;
    } else if (key == "intranode_efficiency_factor") {
      intranode_efficiency_factor_ = std::stod(value);
    } else if (key == "internode_efficiency_factor") {
      internode_efficiency_factor_ = std::stod(value);
    }
    // per_soc_memory_* keys are informational for now; memory comes from
    // q300.txtpb via the performance model.
  }

  file.close();

  if (device_id_layout_ != "row_major_socs_first") {
    std::cerr << "ERROR: Q300ClusterConfig::LoadFromFile - Unsupported "
                 "device_id_layout: "
              << device_id_layout_ << " (only row_major_socs_first supported)"
              << std::endl;
    return false;
  }

  if (num_racks_ < 1) {
    std::cerr << "ERROR: Q300ClusterConfig::LoadFromFile - num_racks must be "
                 ">= 1 (got "
              << num_racks_ << ")" << std::endl;
    return false;
  }

  if (socs_per_rack_ <= 0 || socs_per_card_ <= 0 || cards_per_rack_ <= 0) {
    std::cerr << "ERROR: Q300ClusterConfig::LoadFromFile - socs_per_rack, "
                 "socs_per_card, and cards_per_rack must be > 0."
              << std::endl;
    return false;
  }

  return true;
}

Q300Coord Q300ClusterConfig::DecodeId(int64_t id) const {
  Q300Coord c{};
  const int64_t socs_per_rack =
      static_cast<int64_t>(socs_per_rack_) > 0
          ? static_cast<int64_t>(socs_per_rack_)
          : static_cast<int64_t>(cards_per_rack_) *
                static_cast<int64_t>(socs_per_card_);
  const int64_t socs_per_card = static_cast<int64_t>(socs_per_card_);

  c.rack = static_cast<int>(id / socs_per_rack);
  const int64_t within_rack = id % socs_per_rack;
  c.card_in_rack = static_cast<int>(within_rack / socs_per_card);
  c.soc_in_card = static_cast<int>(within_rack % socs_per_card);
  return c;
}

std::vector<PathComponent> Q300ClusterConfig::PathBetweenDevices(
    int64_t src, int64_t dst) const {
  Q300Coord s = DecodeId(src);
  Q300Coord d = DecodeId(dst);

  if (s.rack == d.rack && s.card_in_rack == d.card_in_rack &&
      s.soc_in_card == d.soc_in_card) {
    return {PathComponent::GPU};
  }
  // Any other pair (intra- OR cross-rack) is one switch hop away through the
  // domain-wide logical crossbar.
  return {PathComponent::GPU, PathComponent::EthSwitch, PathComponent::GPU};
}

double Q300ClusterConfig::StaticLinkBandwidth(PathComponent a,
                                              PathComponent b) const {
  if (IsLink(a, b, PathComponent::GPU, PathComponent::EthSwitch)) {
    return eic_port_bw_gbytes_;
  }
  return 1.0;
}

double Q300ClusterConfig::StaticOversubscription(PathComponent a,
                                                 PathComponent b) const {
  if (IsLink(a, b, PathComponent::GPU, PathComponent::EthSwitch)) {
    return fabric_oversubscription_;
  }
  return 1.0;
}

double Q300ClusterConfig::EffectiveBandwidth(PathComponent a, PathComponent b,
                                             int devices_sharing) const {
  if (devices_sharing < 1) devices_sharing = 1;
  if (IsLink(a, b, PathComponent::GPU, PathComponent::EthSwitch)) {
    // Per-SoC dedicated EIC egress; never shared across replica-group peers.
    return eic_port_bw_gbytes_;
  }
  return 1.0;
}

int Q300ClusterConfig::ComputeDevicesSharingHop(
    PathComponent a, PathComponent b,
    const std::vector<int64_t> &device_ids) const {
  if (IsLink(a, b, PathComponent::GPU, PathComponent::EthSwitch)) {
    return 1;
  }
  (void)device_ids;
  return 1;
}

CommType Q300ClusterConfig::DetermineCommTypeFromPath(
    const std::vector<PathComponent> &path) const {
  for (PathComponent c : path) {
    if (c == PathComponent::EthSwitch) {
      return CommType::ScaleUp;
    }
  }
  return CommType::ScaleUp;
}

CommCostStats Q300ClusterConfig::CalculateCommCost(
    double per_device_comm_volume,
    const stream_executor::DeviceDescription & /*device_info*/,
    const xla::HloInstruction * /*instr*/, uint64_t replica_group_size,
    uint64_t /*num_replica_groups*/, const std::vector<int64_t> &device_ids,
    const std::vector<int> & /*mesh_shape*/,
    const std::string & /*hardware_architecture*/,
    const std::string & /*fallback_device_type*/) {
  if (per_device_comm_volume < 0.0) {
    CHECK_GE(per_device_comm_volume, 0.0)
        << "per_device_comm_volume must be >= 0, got: "
        << per_device_comm_volume;
  }
  if (device_ids.empty()) {
    LOG(FATAL) << "Q300ClusterConfig::CalculateCommCost - device_ids "
                  "cannot be empty";
  }

  const double per_device_comm_vol_gb =
      per_device_comm_volume / (1024.0 * 1024.0 * 1024.0);

  double max_comm_cost_us = 0.0;
  int max_hops = 0;
  std::vector<PathComponent> longest_path;
  // Per-SoC scale-up egress is bounded entirely by the EIC bundle
  // (`eic_port_bw_gbytes` = sum of all PAM4 lanes); there is no
  // multi-rail / multi-port multiplier on top.

  for (size_t i = 0; i < device_ids.size(); ++i) {
    const int64_t src = device_ids[i];
    const int64_t dst = device_ids[(i + 1) % device_ids.size()];
    std::vector<PathComponent> path = PathBetweenDevices(src, dst);
    if (path.size() < 2) continue;

    double bottleneck_gbps = std::numeric_limits<double>::infinity();
    for (size_t h = 0; h + 1 < path.size(); ++h) {
      const int sharing =
          ComputeDevicesSharingHop(path[h], path[h + 1], device_ids);
      const double bw = EffectiveBandwidth(path[h], path[h + 1], sharing);
      if (bw < bottleneck_gbps) bottleneck_gbps = bw;
    }
    if (!std::isfinite(bottleneck_gbps) || bottleneck_gbps <= 0.0) continue;

    const int hop_count = static_cast<int>(path.size() - 1);
    const double hop_latency_us = 1.0;
    const double path_cost_us =
        (per_device_comm_vol_gb / bottleneck_gbps) * 1e6 +
        hop_count * hop_latency_us;

    if (path_cost_us > max_comm_cost_us) {
      max_comm_cost_us = path_cost_us;
      max_hops = hop_count;
      longest_path = path;
    }
  }

  CommType comm_type = CommType::ScaleUp;
  if (!longest_path.empty()) {
    comm_type = DetermineCommTypeFromPath(longest_path);
  }

  // Single-crossbar model: cross-rack and intra-rack are byte-identical, so
  // intranode_efficiency_factor_ applies to every Q300 path.
  const double efficiency = intranode_efficiency_factor_;
  if (efficiency > 0.0 && efficiency < 1.0 && max_comm_cost_us > 0.0) {
    max_comm_cost_us /= efficiency;
  }
  (void)replica_group_size;

  const double intranode_comm_vol_gb = per_device_comm_vol_gb;
  const double internode_comm_vol_gb = 0.0;
  const double total_comm_vol_gb = intranode_comm_vol_gb;
  const double avg_bandwidth_gbps =
      (max_comm_cost_us > 0.0)
          ? per_device_comm_vol_gb / (max_comm_cost_us / 1e6)
          : 0.0;
  const double per_link_cost_us =
      (max_hops > 0) ? max_comm_cost_us / max_hops : 0.0;

  return {comm_type,
          max_comm_cost_us,
          intranode_comm_vol_gb,
          internode_comm_vol_gb,
          avg_bandwidth_gbps,
          avg_bandwidth_gbps,
          total_comm_vol_gb,
          per_device_comm_vol_gb,
          max_hops,
          per_link_cost_us};
}

