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

// Path component types for GPU communication.
//
// The first six entries (GPU..CoreSwitch) are used by GpuClusterConfig and
// must keep their current ordering. The remaining entries (L1Switch..
// OpticalSwitch) are used exclusively by CalciumClusterConfig to model the
// q250 4-level PCIe + L4 Ethernet fabric and the q250l200 rack-scale
// optical fabric. They are unreachable from GpuClusterConfig::PathBetweenDevices
// (which only emits the first six), so existing arch behavior (TPU, B200,
// B300, R200, R576, RCPX) is unaffected.
enum class PathComponent {
    GPU,
    NvSwitch,
    NIC,
    RailSwitch,
    SpineSwitch,
    CoreSwitch,
    L1Switch,       // Calcium q250: PCIe Gen5 x16 SoC <-> L1 switch (intra-pod)
    L2Switch,       // Calcium q250: PCIe Gen6 x32 L1 <-> L2 switch (intra-card)
    L3Switch,       // Calcium q250: PCIe Gen6 x64 L2 <-> L3 switch (intra-server)
    ToRSwitch,      // Calcium: Ethernet RoCE v2 NIC <-> ToR switch (inter-server)
    OpticalSwitch   // Calcium q250l200: rack-wide optical switch (cross-pod intra-rack)
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
        const std::string& fallback_device_type)  = 0;
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
        const std::string& fallback_device_type)  override;
    
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
        const std::string& fallback_device_type)  override;
    
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

// Calcium (q250) physical coordinate: a single SoC's location in the
// server -> card -> L1-pod -> SoC-in-pod hierarchy. Used by
// CalciumClusterConfig::DecodeId and PathBetweenDevices.
//
// q250 (4-level PCIe fabric): rack=0 always; (server, card, l1_pod,
//   soc_in_pod) carry the intra-rack coordinate.
// q250l200 (rack-scale optical fabric): rack varies; (server, card, l1_pod)
//   are derived from pod_in_rack solely for L4 NIC bundling on cross-rack
//   paths. Intra-rack pairs are matched by (rack, server, card, l1_pod) for
//   the same-pod test - see PathBetweenDevices.
struct CalciumCoord {
    int rack;         // 0..N-1   (q250l200: 0..6;  q250: 0)
    int server;       // 0..6     (q250: physical; q250l200: pod_in_rack/24)
    int card;         // 0..7     (q250: physical; q250l200: derived)
    int l1_pod;       // 0..2     (q250: physical; q250l200: derived)
    int soc_in_pod;   // 0..7     (always physical position within pod)
};

// CalciumClusterConfig: 4-level fabric (L1/L2/L3 PCIe + L4 Ethernet RoCE v2)
// for the non-CUDA Calcium accelerator (q250). Models intra-pod, intra-card,
// intra-server, and inter-server collective costs with explicit
// oversubscription and replica-group contention sharing.
class CalciumClusterConfig : public ClusterConfig {
private:
    std::string name_pattern_;

    // L1: PCIe Gen5 x16 SoC <-> L1Switch.
    int    l1_pod_size_;             // 8 SoCs per L1 pod
    double l1_link_bw_gbytes_;       // 64 GB/s UNI
    double l1_oversubscription_;     // 1.0 (non-blocking)

    // L2: PCIe Gen6 x32 L1Switch <-> L2Switch.
    int    l1_pods_per_card_;        // 3 L1 pods per card
    double l2_link_bw_gbytes_;       // 256 GB/s UNI
    double l2_oversubscription_;     // 2.0 (8 SoCs share per L2 uplink)

    // L3: PCIe Gen6 x64 L2Switch <-> L3Switch.
    int    cards_per_server_;        // 8 cards per server
    double l3_link_bw_gbytes_;       // 512 GB/s UNI
    double l3_oversubscription_;     // ~3.0 (24 SoCs share L3 fabric)

    // L4: Ethernet RoCE v2 NIC <-> ToR <-> NIC.
    double host_nic_link_bw_gbytes_; // 800 GB/s aggregate
    double l4_per_card_egress_gbytes_; // 100 GB/s per card (the 5.12:1 cliff)
    double tor_switch_uplink_gbytes_;  // 800 GB/s

    // q250l200 optical fabric (set when optical_port_bw_gbytes > 0). When
    // is_l200_ is true, the L1/L2/L3 PCIe fields are NOT used by intra-rack
    // path-finding; only optical_port + rack switch + L4 RoCE are emitted.
    bool   is_l200_;
    double optical_port_bw_gbytes_;            // 640 GB/s per-SoC optical port
    double optical_switch_oversubscription_;   // typically 1.0 (non-blocking)
    int    socs_per_rack_;                     // 1344
    int    pods_per_rack_;                     // 168 = 1344/8
    int    servers_per_rack_;                  // 7

    // Efficiency factors.
    double intranode_efficiency_factor_;
    double internode_efficiency_factor_;

    // Device-id layout convention. Only "row_major_socs_first" supported today.
    std::string device_id_layout_;

public:
    CalciumClusterConfig();
    virtual ~CalciumClusterConfig() = default;

    // ClusterConfig interface implementation.
    std::string GetNamePattern() const override;
    bool LoadFromFile(const std::string& config_file_path) override;
    CommCostStats CalculateCommCost(double per_device_comm_volume,
        const stream_executor::DeviceDescription& device_info,
        const xla::HloInstruction* instr,
        uint64_t replica_group_size,
        uint64_t num_replica_groups,
        const std::vector<int64_t>& device_ids,
        const std::vector<int>& mesh_shape,
        const std::string& hardware_architecture,
        const std::string& fallback_device_type) override;

    // Calcium-specific public interface.
    // Decode a flat device ID into physical coordinates.
    // Layout: id = server*192 + card*24 + l1_pod*8 + soc_in_pod.
    CalciumCoord DecodeId(int64_t id) const;

    // Return the 2/4/6/10-hop path between two SoCs based on the LCA of
    // their physical coordinates.
    std::vector<PathComponent> PathBetweenDevices(int64_t src, int64_t dst) const;

    // Static link bandwidth (GB/s) for a single hop (src, dst). Pure topology.
    // Used for --dump_path_trace parity check vs. the rack-level trace.
    double StaticLinkBandwidth(PathComponent a, PathComponent b) const;

    // Static topological oversubscription factor for a hop. Informational /
    // diagnostic: a structural ratio of (downstream-leaf demand) / (upstream
    // link capacity). NOT used directly by EffectiveBandwidth - that quantity
    // is fully determined by (link_bw, devices_sharing).
    double StaticOversubscription(PathComponent a, PathComponent b) const;

    // Effective per-SoC bandwidth on a hop, accounting for replica-group
    // sharing on the upstream link AND the per-SoC leaf-link cap (each SoC's
    // own dedicated egress is at most l1_link_bw_gbytes_). devices_sharing is
    // the count of SoCs in the replica group that traverse the same switch
    // uplink at this hop. Returns min(leaf_bw, link_bw / devices_sharing).
    double EffectiveBandwidth(PathComponent a, PathComponent b,
                              int devices_sharing) const;

    // For each hop in the path, compute how many SoCs in `device_ids`
    // share the bottleneck switch uplink.
    int ComputeDevicesSharingHop(PathComponent a, PathComponent b,
                                 const std::vector<int64_t>& device_ids) const;

    // Step 8 (PP): inter-stage handoff cost between rank-0 SoC of stage 0
    // and rank-0 SoC of stage 1. Uses PathBetweenDevices + EffectiveBandwidth.
    // P2P (1-to-1), so sharing=1 on every hop.
    double CalculatePipelineHandoffCost(int64_t activation_bytes,
                                        int devices_per_stage) const;

    // Determine CommType from a Calcium path. Used by CSV export.
    CommType DetermineCommTypeFromPath(const std::vector<PathComponent>& path) const;

    // Diagnostic getters.
    double GetIntranodeEfficiencyFactor() const { return intranode_efficiency_factor_; }
    double GetInternodeEfficiencyFactor() const { return internode_efficiency_factor_; }
};

// Factory function to create cluster config based on device type
std::unique_ptr<ClusterConfig> CreateClusterConfig(const std::string& device_type);

// Helper function to get cluster config by name pattern.
// configs_dir: directory containing device_name.config; when empty, uses
// path xla/service/gpu/model/configs/ (legacy, for callers that do not pass it).
std::unique_ptr<ClusterConfig> GetClusterConfigByName(
    const std::string& device_name,
    const std::string& configs_dir);

#endif // CLUSTER_CONFIG_H