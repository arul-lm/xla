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

// Path component types for inter-device communication.
//
// The first six entries (GPU..CoreSwitch) are used by GpuClusterConfig and
// must keep their current ordering. The remaining entries (L1Switch..
// OpticalSpine) are used exclusively by CalciumClusterConfig to model the
// q250 4-level PCIe + L4 Ethernet fabric and the q250l200 hybrid optical
// fabric (intra-rack + datacenter-scale optical spine). They are unreachable
// from GpuClusterConfig::PathBetweenDevices (which only emits the first six),
// so existing arch behavior (TPU, B200, B300, R200, R576, RCPX) is unaffected.
//
// PathComponent::GPU is the cross-arch enum and is used for the q250/q250l200
// "SoC" device too. Calcium docs and user-facing artifacts render this as
// "SOC" (the device is a System-on-Chip, not a GPU), but the enum stays
// shared because the path-finding machinery is identical.
enum class PathComponent {
    GPU,            // Calcium: SoC (System-on-Chip)
    NvSwitch,
    NIC,
    RailSwitch,
    SpineSwitch,
    CoreSwitch,
    L1Switch,       // Calcium q250: PCIe Gen5 x16 SoC <-> L1 switch (intra-pod)
    L2Switch,       // Calcium q250: PCIe Gen6 x32 L1 <-> L2 switch (intra-card)
    L3Switch,       // Calcium q250: PCIe Gen6 x64 L2 <-> L3 switch (intra-server)
    ToRSwitch,      // Calcium: Ethernet RoCE v2 NIC <-> ToR switch (inter-server)
    OpticalSwitch,  // Calcium q250l200: rack-wide optical switch (cross-pod intra-rack)
    OpticalSpine,   // Calcium q250l200: cross-rack optical aggregation (datacenter-scale)
    EthSwitch       // Q300: flat ESUN 1.0 single-tier rack switch (intra-rack only)
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

// Per-boundary pipeline handoff cost statistics. Produced by
// CalciumClusterConfig::CalculatePipelineHandoffCosts. All times are in
// microseconds and include 1-us-per-hop latency adders.
//
// Two views are exposed because they have different physical meanings:
//
//  - "raw" values are the full handoff cost on the critical path of a
//    single microbatch (no overlap can hide any of it, because microbatch m
//    cannot start stage k+1 until its own handoff completes). T_first /
//    pipeline-fill / single-microbatch traversal cost uses these.
//
//  - "effective" values are the raw values multiplied by
//    (1 - pipeline_comm_overlap_factor). They model how much of each
//    handoff is hidden behind the NEXT microbatch's compute on the
//    previous stage in steady state. T_step (cadence) and steady-state
//    throughput use these.
//
// `pipeline_comm_overlap_factor` is distinct from the within-stage
// compute/communication `overlap_factor` used by
// CalculateInstructionLevelOverlap (which controls how t_stage_us itself
// is computed from the per-instruction compute and comm times). The
// pipeline factor controls only inter-stage handoff hiding.
//
// For pipeline_comm_overlap_factor == 0.0 the two views are identical
// (byte-stable pre-Step-8b behavior).
struct PipelineHandoffStats {
  // Effective values (after applying (1 - pipeline_comm_overlap_factor)).
  double max_us = 0.0;               // Slowest boundary; drives T_step.
  double sum_us = 0.0;               // Sum across boundaries; convenience only.
  double avg_us = 0.0;               // sum_us / num_boundaries (diagnostic).

  // Raw values (no overlap applied). Drive T_first / bubble / per-microbatch
  // pipeline traversal time.
  double raw_max_us = 0.0;
  double raw_sum_us = 0.0;

  int    num_boundaries = 0;              // num_pipeline_stages - 1.
  double pipeline_comm_overlap_factor = 0.0;  // Factor actually applied [0,1].
  std::vector<double> per_boundary_us;    // Effective per-boundary cost.
  std::vector<double> per_boundary_raw_us;// Raw per-boundary cost.
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
//
// Step 9 adds an optional hierarchical AllReduce cost model toggled
// per-call via SetHierarchicalAllReduceEnabled (driven by the FFI / CLI
// `hierarchical_allreduce_enabled` parameter, default OFF). When ON it
// short-circuits the flat worst-pair AR loop with a recursive
// {Pod, Card, Server, Rack} decomposition that charges intra-tier volume
// at the local tier's bandwidth and only the T/k cross-tier shard at the
// slow tier. Every other opcode and every other arch (B200/R200/R576/TPU)
// is untouched.
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
    // path-finding; only optical_port + rack switch (+ optional cross-rack
    // optical spine) are emitted.
    bool   is_l200_;
    double optical_port_bw_gbytes_;            // 640 GB/s per-SoC optical port
    double optical_switch_oversubscription_;   // typically 1.0 (non-blocking)
    int    socs_per_rack_;                     // 1344
    int    pods_per_rack_;                     // 168 = 1344/8
    int    servers_per_rack_;                  // 7

    // Multi-rack fabric (q250l200 only). N == 1 (default) preserves the
    // existing single-rack behavior: cross-rack PathBetweenDevices branches
    // are unreachable because every legal device id decodes to rack == 0.
    // N > 1 enables the cross-rack optical-spine path
    //   SoC -> OpticalSwitch (rack) -> OpticalSpine -> OpticalSwitch (rack) -> SoC
    // which stays in the scale-up domain (no ToRSwitch, no L4 RoCE) and is
    // tunable via optical_spine_uplink_gbytes_ + optical_spine_oversubscription_.
    int    num_racks_;                         // 1 = single-rack (byte-stable)
    double optical_spine_uplink_gbytes_;       // per rack-to-spine optical uplink
    double optical_spine_oversubscription_;    // 1.0 = non-blocking spine

    // Efficiency factors.
    double intranode_efficiency_factor_;
    double internode_efficiency_factor_;

    // Device-id layout convention. Only "row_major_socs_first" supported today.
    std::string device_id_layout_;

    // Step 9: hierarchical AllReduce gating. Default off keeps the flat
    // worst-pair AR cost path byte-stable. Toggled exclusively by callers
    // via SetHierarchicalAllReduceEnabled() (used by the FFI parameter
    // `hierarchical_allreduce_enabled` on
    // analytical_latency_calculator_run_with_pipeline and the matching
    // `--hierarchical-allreduce-enabled` CLI flag). Intentionally NOT a
    // config-file key. Affects only kAllReduce / kAllReduceStart opcodes;
    // every other collective (AllGather, ReduceScatter, etc.) keeps the
    // existing flat cost model regardless of this flag.
    bool hierarchical_allreduce_enabled_;

public:
    CalciumClusterConfig();
    virtual ~CalciumClusterConfig() = default;

    // Internal tier used by the hierarchical AllReduce decomposition.
    // Public so unit tests / smokes can drive the helpers directly without
    // friend declarations.
    enum class Tier { Pod, Card, Server, Rack };

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
    //
    // Kept for ABI/API compatibility with pre-Step-8b callers. Equivalent to
    // CalculatePipelineHandoffCosts(activation_bytes, devices_per_stage,
    //                               /*num_pipeline_stages=*/2,
    //                               /*pipeline_comm_overlap_factor=*/0.0).max_us
    // for any layout where every stage boundary lands on the same path tier
    // (i.e. when devices_per_stage divides cleanly into the q250 hierarchy).
    // Otherwise it returns the cost of the FIRST boundary only, which can
    // under-estimate cadence; new code should call the per-boundary variant.
    double CalculatePipelineHandoffCost(int64_t activation_bytes,
                                        int devices_per_stage) const;

    // Step 8b (PP): per-boundary handoff cost. Computes one handoff for each
    // of the (num_pipeline_stages - 1) stage boundaries, walking
    //   (k, k+1) -> (src = k * devices_per_stage, dst = (k+1) * devices_per_stage)
    // for k in [0, S-2]. Each boundary uses sharing=1 (P2P) and the leaf-
    // bandwidth-aware EffectiveBandwidth on its own path.
    //
    // pipeline_comm_overlap_factor in [0.0, 1.0] models inter-stage
    // compute/handoff overlap: the visible per-boundary cost is multiplied
    // by (1 - pipeline_comm_overlap_factor). 0.0 = no overlap (legacy);
    // 0.5 = typical 1F1B-style overlap with the next stage's compute;
    // 1.0 = handoff fully hidden. This is distinct from the within-stage
    // compute/comm `overlap_factor` used by CalculateInstructionLevelOverlap
    // - they apply at different layers of the model (intra-stage instruction
    // overlap vs inter-stage handoff overlap) and compose multiplicatively.
    //
    // Returns max/sum/avg over the S-1 boundaries plus the per-boundary
    // vector for diagnostics. Aggregation semantics:
    //   T_step  uses .max_us  (slowest boundary drives steady-state cadence)
    //   T_first uses .raw_sum_us (fill phase pays each boundary in full)
    //   bubble  uses .raw_sum_us
    PipelineHandoffStats CalculatePipelineHandoffCosts(
        int64_t activation_bytes, int devices_per_stage,
        int num_pipeline_stages,
        double pipeline_comm_overlap_factor) const;

    // Determine CommType from a Calcium path. Used by CSV export.
    CommType DetermineCommTypeFromPath(const std::vector<PathComponent>& path) const;

    // Diagnostic getters.
    double GetIntranodeEfficiencyFactor() const { return intranode_efficiency_factor_; }
    double GetInternodeEfficiencyFactor() const { return internode_efficiency_factor_; }
    bool   GetHierarchicalAllReduceEnabled() const {
        return hierarchical_allreduce_enabled_;
    }
    // Caller-side toggle for the hierarchical AllReduce cost model. The
    // FFI / CLI is the only way to enable Step 9 - there is no .config
    // key. Default state is OFF (built-in C++ default in the constructor).
    // Idempotent.
    void   SetHierarchicalAllReduceEnabled(bool enabled) {
        hierarchical_allreduce_enabled_ = enabled;
    }

    // ----- Step 9: hierarchical AllReduce helpers -----
    //
    // These are only consulted when hierarchical_allreduce_enabled_ is true
    // AND the opcode is kAllReduce / kAllReduceStart. They are public solely
    // for testability; production callers should go through CalculateCommCost.

    // Partition `device_ids` at the given tier. Two ids land in the same
    // subgroup iff they share the tier-defining coordinates (Pod groups by
    // (rack, server, card, l1_pod), Card by (rack, server, card), Server by
    // (rack, server), Rack by (rack)).
    std::vector<std::vector<int64_t>> PartitionAtTier(
        const std::vector<int64_t>& device_ids, Tier tier) const;

    // Auto-pick the tier list for a replica group based on the span of the
    // decoded coordinates. Returns the smallest tier list that captures the
    // actual cross-tier structure of `device_ids`. Returns {} when the whole
    // group fits in a single pod (no decomposition is profitable).
    std::vector<Tier> AutoSelectTiers(
        const std::vector<int64_t>& device_ids) const;

    // Single-stage RS or AG cost (microseconds) on a homogeneous subgroup.
    // Uses worst-pair semantics inside the subgroup (matching flat AR's
    // per-stage charging convention).
    //
    // `parallel_rails` is the cumulative product of enclosing-stage subgroup
    // sizes (1 at the top-level call). It inflates the participant count on
    // shared upstream hops to model the physical fact that all `parallel_rails`
    // outer-group peers send simultaneously over each shared upstream link.
    // The leaf hop (SoC<->L1, SoC<->SoC interposer, SoC<->OpticalSwitch,
    // SoC<->NIC) is unaffected because it is a per-SoC dedicated egress, not
    // a shared link.
    double SubgroupRsAgCostUs(const std::vector<int64_t>& subgroup,
                              double per_device_bytes,
                              int parallel_rails) const;

    // Flat AllReduce cost (microseconds) over `device_ids`, with the same
    // worst-pair logic the legacy CalculateCommCost loop uses, but
    // additionally inflated by `parallel_rails` on shared upstream hops.
    // Top-level callers pass parallel_rails=1, which preserves byte-stable
    // flat-AR output.
    double FlatAllReduceCostUs(const std::vector<int64_t>& device_ids,
                               double per_device_bytes,
                               int parallel_rails) const;

    // Recursive hierarchical AllReduce cost (microseconds) over `device_ids`,
    // partitioning at `tiers[0]` first and recursing on the subgroup
    // representatives with the (T/k)-shard tensor. Falls back to
    // FlatAllReduceCostUs when `tiers` is empty (base case) or when the
    // current tier yields a single subgroup (no decomposition possible at
    // this tier; descend).
    //
    // NOTE: takes the TENSOR size T (not the per-device AR volume
    // 2(N-1)/N * T). The single conversion happens at the top-level call
    // site in CalculateCommCost so the plan's stage-volume formulas
    // (T*(k-1)/k for RS/AG, T/k for the recursive cross-subgroup AR) can be
    // expressed directly.
    //
    // Returns the sum of stage costs (RS_intra + AR_inter + AG_intra). The
    // caller is responsible for applying the outer efficiency factor.
    double HierarchicalAllReduceCostUs(
        const std::vector<int64_t>& device_ids,
        double tensor_bytes,
        const std::vector<Tier>& tiers,
        int parallel_rails) const;
};

// Q300 physical coordinate: flat (rack, card_in_rack, soc_in_card).
// v1 is single-rack only (rack == 0 always).
struct Q300Coord {
    int rack;          // 0 in v1
    int card_in_rack;  // 0..71
    int soc_in_card;   // 0..7
};

// Q300ClusterConfig: flat 1-tier ESUN Ethernet scale-up fabric for the Q300
// accelerator (hw_arch tokens q300 / q300l200). The whole scale-up domain is
// modeled as a single non-blocking crossbar of radix
// num_racks * socs_per_rack: any two distinct SoCs in the domain are exactly
// one switch hop apart (SoC -> EthSwitch -> SoC), regardless of rack.
// CommType is always ScaleUp.
class Q300ClusterConfig : public ClusterConfig {
private:
    std::string name_pattern_;
    int socs_per_card_;
    int cards_per_rack_;
    int socs_per_rack_;
    int num_racks_;
    double eic_port_bw_gbytes_;
    double fabric_oversubscription_;
    int parallel_rails_;
    double intranode_efficiency_factor_;
    double internode_efficiency_factor_;   // parsed but unused in v1.5
    std::string device_id_layout_;

public:
    Q300ClusterConfig();
    virtual ~Q300ClusterConfig() = default;

    std::string GetNamePattern() const override;
    bool LoadFromFile(const std::string& config_file_path) override;
    CommCostStats CalculateCommCost(double per_device_comm_volume,
        const stream_executor::DeviceDescription& device_info,
        const xla::HloInstruction* instr, uint64_t replica_group_size,
        uint64_t num_replica_groups,
        const std::vector<int64_t>& device_ids,
        const std::vector<int>& mesh_shape,
        const std::string& hardware_architecture,
        const std::string& fallback_device_type) override;

    Q300Coord DecodeId(int64_t id) const;
    std::vector<PathComponent> PathBetweenDevices(int64_t src, int64_t dst) const;
    double StaticLinkBandwidth(PathComponent a, PathComponent b) const;
    double StaticOversubscription(PathComponent a, PathComponent b) const;
    double EffectiveBandwidth(PathComponent a, PathComponent b,
                              int devices_sharing) const;
    int ComputeDevicesSharingHop(PathComponent a, PathComponent b,
                                 const std::vector<int64_t>& device_ids) const;
    CommType DetermineCommTypeFromPath(
        const std::vector<PathComponent>& path) const;

    double GetIntranodeEfficiencyFactor() const {
        return intranode_efficiency_factor_;
    }
    int GetParallelRails() const { return parallel_rails_; }
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