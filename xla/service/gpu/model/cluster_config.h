#ifndef CLUSTER_CONFIG_H
#define CLUSTER_CONFIG_H

#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace xla {
    class HloInstruction;
}

namespace stream_executor {
    class DeviceDescription;
}

// Forward declaration for CommCostStats
struct CommCostStats;

// Path component types for GPU communication
enum class PathComponent {
    GPU,
    NvSwitch,
    NIC,
    RailSwitch,
    SpineSwitch,
    CoreSwitch
};

// Communication type enum for node-level analysis
enum class CommType {
  ScaleUp,      // GPU clusters: only GPUs and NvSwitches
  Rail,         // GPU clusters: GPUs, NICs, Rails
  RailOffset,   // GPU clusters: GPUs, NvSwitches, Rails, NICs
  ScaleOut      // GPU clusters: GPUs, NICs, Rails, Spines, Cores
};

// Structure to hold intra-node configuration
struct IntraNodeConfig {
    int size;           // Number of GPUs per node
    double bandwidth_gbps;  // Total bandwidth in GB/s shared between all GPUs in the node
    double efficiency_factor;  // Efficiency factor for collective operations (0.0 to 1.0)
    bool reassign_bandwidth;  // Whether to reassign bandwidth for this device type
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
  double per_device_comm_vol_gb;  // Per-device communication volume in GB
  int num_hops;              // Number of hops for communication
  double per_link_cost_us;   // Cost per link in microseconds
};

// Base class for cluster configurations
class ClusterConfig {
public:
    virtual ~ClusterConfig() = default;
    
    // Pure virtual methods that must be implemented by derived classes
    virtual std::string GetNamePattern() const = 0;
    virtual bool LoadFromFile(const std::string& config_file_path) = 0;
    virtual CommCostStats CalculateCommCost(double per_device_comm_volume,
        const stream_executor::DeviceDescription& tpu_device_info,
        const xla::HloInstruction* instr, uint64_t replica_group_size,
        uint64_t num_replica_groups,
        const std::vector<int64_t>& device_ids,
        const std::vector<int>& mesh_shape,
        const std::string& hardware_architecture,
        const std::string& fallback_device_type = "")  = 0;
};

// TPU Cluster Configuration
class TpuClusterConfig : public ClusterConfig {
private:
    std::string name_pattern_;
    double intranode_bandwidth_gbps_;
    double internode_bandwidth_gbps_;
    double intranode_efficiency_factor_;
    double internode_efficiency_factor_;
    bool reassign_bandwidth_;
    std::vector<int> intranode_mesh_shape_;
    std::vector<int> internode_mesh_shape_;
    int links_per_chip_;
    
public:
    TpuClusterConfig();
    virtual ~TpuClusterConfig() = default;
    
    // ClusterConfig interface implementation
    std::string GetNamePattern() const override;
    bool LoadFromFile(const std::string& config_file_path) override;
    CommCostStats CalculateCommCost(double per_device_comm_volume,
        const stream_executor::DeviceDescription& tpu_device_info,
        const xla::HloInstruction* instr, uint64_t replica_group_size,
        uint64_t num_replica_groups,
        const std::vector<int64_t>& device_ids,
        const std::vector<int>& mesh_shape,
        const std::string& hardware_architecture,
        const std::string& fallback_device_type = "")  override;
    
    // TPU-specific methods
    void SetConfig(const std::string& name_pattern,
                   double intranode_bandwidth_gbps,
                   double internode_bandwidth_gbps,
                   double intranode_efficiency_factor,
                   double internode_efficiency_factor,
                   bool reassign_bandwidth,
                   const std::vector<int>& intranode_mesh_shape,
                   const std::vector<int>& internode_mesh_shape);
    
    // TPU-specific getters
    std::vector<int> GetIntranodeMeshShape() const;
    std::vector<int> GetInternodeMeshShape() const;
    int GetLinksPerChip() const;
    double GetIntranodeBandwidthGbps() const;
    double GetInternodeBandwidthGbps() const;
    double GetIntranodeEfficiencyFactor() const;
    double GetInternodeEfficiencyFactor() const;
    bool GetReassignBandwidth() const;
};

// Structure to hold scaleup domain configuration
struct ScaleupDomain {
    int scaleup_switch_count;
    int scaleup_device_count;
    std::vector<int> scaleup_fabric_link_bw;  // [port_count, bandwidth_gbytes]
    std::string scaleup_fabric_topology;
    double per_xpu_bw_gbytes;
};

// Structure to hold scaleout domain configuration  
struct ScaleoutDomain {
    int leaf_switches_per_rail;
    int spine_switches_per_slg;
    int core_switches_per_cg;
    double pcie_bw;
    std::vector<int> nic_to_leaf_switch_bw;  // [port_count, bandwidth_gbytes]
    std::vector<int> leaf_to_spine_bw;       // [port_count, bandwidth_gbytes]
    std::vector<int> spine_to_core_bw;       // [port_count, bandwidth_gbytes]
    int scalable_unit_count;
    int rack_count;
    int tray_count;
    int unit_count;
    std::vector<int> subscription_ratio;
    std::string fat_tree_type;
};

// GPU Cluster Configuration
class GpuClusterConfig : public ClusterConfig {
private:
    std::string name_pattern_;
    double intranode_efficiency_factor_;
    double internode_efficiency_factor_;
    bool reassign_bandwidth_;
    
    int scalable_unit_count_;
    int compute_rack_count_;
    int compute_trays_per_rack_;
    int compute_units_per_tray_;
    int nic_per_tray_;
    double nic_speed_gbytes_;
    double scaleup_fabric_link_bw_gbytes_;
    int scaleup_fabric_port_count_;
    int scaleup_switch_port_count_;
    int scaleup_switch_latency_;
    std::string scaleup_fabric_topology_;
    double pcie_link_bw_;
    int pcie_link_count_;
    std::string scaleout_fabric_topology_;
    int mgmt_rack_count_;
    int leaf_switch_port_count_;
    int spine_switch_port_count_;
    int core_switch_port_count_;
    std::string scaleout_subscription_;
    
    // Domain configurations built from the above parameters
    ScaleupDomain scaleup_domain_;
    ScaleoutDomain scaleout_domain_;
    
public:
    GpuClusterConfig();
    virtual ~GpuClusterConfig() = default;
    
    // ClusterConfig interface implementation
    std::string GetNamePattern() const override;
    bool LoadFromFile(const std::string& config_file_path) override;
    CommCostStats CalculateCommCost(double per_device_comm_volume,
        const stream_executor::DeviceDescription& tpu_device_info,
        const xla::HloInstruction* instr, uint64_t replica_group_size,
        uint64_t num_replica_groups,
        const std::vector<int64_t>& device_ids,
        const std::vector<int>& mesh_shape,
        const std::string& hardware_architecture,
        const std::string& fallback_device_type = "")  override;
    
    // GPU-specific methods
    void SetConfig(const std::string& name_pattern,
                   double intranode_efficiency_factor,
                   double internode_efficiency_factor,
                   bool reassign_bandwidth,
                   int scalable_unit_count,
                   int compute_rack_count,
                   int compute_trays_per_rack,
                   int compute_units_per_tray,
                   int nic_per_tray,
                   double nic_speed_gbytes,
                   double scaleup_fabric_link_bw_gbytes,
                   int scaleup_fabric_port_count,
                   int scaleup_switch_port_count,
                   int scaleup_switch_latency,
                   const std::string& scaleup_fabric_topology,
                   double pcie_link_bw,
                   int pcie_link_count,
                   const std::string& scaleout_fabric_topology,
                   int mgmt_rack_count,
                   int leaf_switch_port_count,
                   int spine_switch_port_count,
                   int core_switch_port_count,
                   const std::string& scaleout_subscription);
    
    // GPU-specific getters
    double GetIntranodeEfficiencyFactor() const;
    double GetInternodeEfficiencyFactor() const;
    bool GetReassignBandwidth() const;
    
    // Domain configuration getters
    const ScaleupDomain& GetScaleupDomain() const;
    const ScaleoutDomain& GetScaleoutDomain() const;
    
    // Build domain configurations from loaded parameters
    void BuildDomains();
    
    // Path calculation helper methods
    std::pair<int, int> GpuIdToScaleupCoordinates(int gpu_id) const;
    std::tuple<int, int, int, int> GpuIdToScaleoutCoordinates(int gpu_id) const;
    std::vector<PathComponent> PathBetweenDevices(int src_device_id, int dest_device_id) const;
    std::vector<double> PathToBandwidth(const std::vector<PathComponent>& path) const;
    CommType DetermineCommTypeFromPath(const std::vector<PathComponent>& path) const;
    std::pair<int, int> CountHopsByDomain(const std::vector<PathComponent>& path) const;
};

// Factory function to create cluster config based on device type
std::unique_ptr<ClusterConfig> CreateClusterConfig(const std::string& device_type);

// Helper function to get cluster config by name pattern
std::unique_ptr<ClusterConfig> GetClusterConfigByName(const std::string& device_name);

#endif // CLUSTER_CONFIG_H