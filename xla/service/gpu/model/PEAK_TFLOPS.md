# Peak matrix (tensor core) TFLOPS

Peak TFLOPS from the GPU performance model (matrix unit, FMA):

**Formula:** `peak_ops_per_ns = clock_rate_ghz × ops_per_clock × 2 × core_count × units_per_core`  
**Peak TFLOPS = peak_ops_per_ns / 1000**

Data source: `target_config/specs/*.txtpb` matrix_unit_description (key 16 = BF16, key 20 = FP8 E4M3, key 32 = FP4 / F4E2M1FN).

## Hardware spec summary (single table)

Dense tensor-core **PFLOPS** match the matrix-unit formula in each row’s **device spec** (same rounding as the per-GPU table later in this doc). **Cluster** fields come from `xla/service/gpu/model/configs/{arch}.config`. **All link speeds in the table are per-link unidirectional** line rates from the config (`scaleup_fabric_link_bw_gbytes`, `nic_speed_gbytes`).

| Config (`hw_arch`) | BF16 dense (PFLOPS) | FP8 dense (PFLOPS) | FP4 dense (PFLOPS) | Scale-up domain (GPUs) | Scale-up aggregate (GB/s) | Scale-up link uni-dir (GB/s) | Scale-out aggregate / SU (GB/s) | Scale-out NIC link uni-dir (GB/s) | HBM BW (TB/s) | HBM per GPU (GiB) |
|--------------------|--------------------:|-------------------:|-------------------:|-----------------------:|--------------------------:|-----------------------------:|--------------------------------:|----------------------------------:|--------------:|------------------:|
| `b200` | 2.25 | 4.5 | 9 | 72 | 900 | 50 | 28,800 | 50 | 7.67 | 178 |
| `b200l200` | 2.25 | 4.5 | 9 | 1,152 | 3,600 | 50 | 460,800 | 50 | 7.67 | 178 |
| `b300` | 3.5 | 7 | 14 | 72 | 900 | 50 | 57,600 | 100 | 7.94 | 288 |
| `b300l200` | 3.5 | 7 | 14 | 1,152 | 3,600 | 50 | 921,600 | 100 | 7.94 | 288 |
| `r200` | 4 | 17.5 | 25 | 72 | 1,800 | 100 | 57,600 | 100 | 22 | 288 |
| `r200l200` | 4 | 17.5 | 25 | 1,152 | 7,800 | 100 | 921,600 | 100 | 22 | 288 |
| `r576` | 8 | 35 | 50 | 576 | 1,800 | 100 | 57,600 | 100 | 53 | 1,024 |
| `r576l200` | 8 | 35 | 50 | 1,152 | 7,200 | 100 | 921,600 | 100 | 53 | 1,024 |

**Definitions:** **Scale-up aggregate** = `scaleup_fabric_port_count × scaleup_fabric_link_bw_gbytes` (all2all fabric model in `cluster_config.cc`). **Scale-out aggregate / SU** = `compute_rack_count × compute_trays_per_rack × nic_per_tray × nic_speed_gbytes`. **HBM** from `device_memory_size` and `memory_bandwidth` in the matching `*.txtpb` (GiB / TB/s rounded like the tables below).

## How peak_matrix_ops is used in compute_time

For **Rubin** (r200, r200l200, r576, r576l200) and **Blackwell** (b200, b300, etc.) devices, the performance model uses `ComputeTimeFromPeakMatrixOps` in `gpu_performance_model.cc`:

1. **Lookup:** `peak_ops_per_ns = CalculatePeakMatrixOpsPerNs(gpu_device_info, dtype)`  
   - Reads `matrix_unit_description.rate_infos[dtype]` from the spec (e.g. key 32 for F4E2M1FN).  
   - Implements: `peak_ops_per_ns = clock_rate_ghz × ops_per_clock × 2 × core_count × units_per_core` (FMA = 2 ops).  
   - Implemented in `GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs` in `gpu_performance_model_base.cc`.

2. **Sparse boost:** `peak_ops_per_ns = peak_ops_per_ns * 2`  
   - So the effective peak used for compute time is **2×** the dense value from the spec (sparse TFLOPS).

3. **Utilization (saturation):** `effective_ops_per_ns = peak_ops_per_ns * utilization`, with utilization in [0.1, 0.8] from a saturation model based on problem size (flops).

4. **Compute time:** `compute_time_ns = flops / effective_ops_per_ns`.

So the **spec stores the dense peak** (ops/ns); the model then applies the 2× sparse boost and the utilization curve to get compute time. To target a specific **sparse** peak (e.g. 50 PFLOPS), set the spec so that the dense peak is half that (e.g. 25 PFLOPS).

## Per-GPU specs by `hw_arch` (`.txtpb`)

Dense tensor-core TFLOPS below follow the matrix-unit formula in the spec (same rounding as the summary table). The analytical model then applies an extra **2× sparse boost** on compute time (see above). **`*l200` device specs** (`b200l200.txtpb`, etc.) match the non-l200 **same chip** for HBM, BW, and matrix rates; only the **cluster `.config`** differs.

| hw_arch | Spec file | HBM (GiB) | HBM BW (TB/s) | SM cores | BF16 dense (PFLOPS) | FP8 dense (PFLOPS) | FP4 dense (PFLOPS) |
|---------|-----------|-----------|---------------|----------|---------------------|--------------------|---------------------|
| b200 | `b200.txtpb` | ~178 | ~7.67 | 148 | ~2.25 | ~4.5 | ~9 |
| b200l200 | `b200l200.txtpb` | ~178 | ~7.67 | 148 | ~2.25 | ~4.5 | ~9 |
| b300 | `b300.txtpb` | ~288 | ~7.94 | 158 | ~3.5 | ~7 | ~14 |
| b300l200 | `b300l200.txtpb` | ~288 | ~7.94 | 158 | ~3.5 | ~7 | ~14 |
| r200 | `r200.txtpb` | ~288 | ~22 | 148 | ~4 | ~17.5 | ~25 |
| r200l200 | `r200l200.txtpb` | ~288 | ~22 | 148 | ~4 | ~17.5 | ~25 |
| r576 | `r576.txtpb` | ~1,024 | ~53 | 148 | ~8 | ~35 | ~50 |
| r576l200 | `r576l200.txtpb` | ~1,024 | ~53 | 148 | ~8 | ~35 | ~50 |

**r576** is Rubin-class like **r200** but with **2×** `matrix_unit_description.ops_per_clock` vs `r200.txtpb` (dense FLOPS doubled in the spec).

### r200, b200, b300 — BF16, FP8, FP4 (dense and sparse)

Values aligned to vendor-published dense TFLOPS for **r200 / b200 / b300**. Sparse = 2× dense from the **model** sparse boost (on top of spec).

| Device | BF16 (key 16) | FP8 E4M3 (key 20) dense | FP8 sparse | FP4 (key 32) dense | FP4 sparse |
|--------|---------------|--------------------------|------------|--------------------|------------|
| r200   | 4,000         | 17,500                   | 35,000     | 25,000             | 50,000     |
| r576   | 8,000         | 35,000                   | 70,000     | 50,000             | 100,000    |
| b200   | 2,250         | 4,500                    | 9,000      | 9,000              | 18,000     |
| b300   | 3,500         | 7,000                    | 14,000     | 14,000             | 28,000     |

- **r200 (Rubin):** BF16 4 PFLOPS, FP8 17.5 PFLOPS, NVFP4 25 PFLOPS dense (before the model’s extra 2× sparse path).
- **r576:** HBM in `r576.txtpb` is **1 TiB / 53 TB/s** (vs r200’s ~288 GiB / ~22 TB/s); **2×** dense matrix peaks vs r200 (see `r576.txtpb`).
- **b200 (Blackwell):** Per [Spheron B300 guide](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/) (B200 column).
- **b300 (Blackwell Ultra):** Per same guide (B300 column).

**`*l200` `.txtpb`** files match their base chip for per-GPU numbers.

---

## Base `hw_arch` vs `*l200` counterpart (cluster `.config`)

Per-GPU `.txtpb` is unchanged between `b200` and `b200l200` (etc.). Differences are **only** in `configs/*.config`: larger scale-up domain, more scale-up fabric ports, more trays/racks per scalable unit, a `scalable_unit_count` that may differ from the paired base config (see each file), and more NICs per tray where applicable (per-SU scale-out grows with tray count × NICs per tray).

**`scalable_unit_count`** (in each `.config`) is the number of **scale-out pods** (scalable units) in the **modeled cluster**. It is used when mapping a global GPU id to “which SU / rack / tray / GPU” (`GpuIdToScaleoutCoordinates` in `cluster_config.cc`) and when building scale-out switch counts. It does **not** set the number of racks per SU by itself; **racks per SU** is `compute_rack_count`, **trays per rack** is `compute_trays_per_rack`. Changing **`scalable_unit_count`** changes how many such pods exist end-to-end (and thus total GPUs in the full cluster story), not the internal layout of one pod.

**Scale-out NIC column:** `nic_speed_gbytes` from the config — **one NIC port’s unidirectional line rate** (GB/s). The model also stores this in `scaleout_domain_.nic_to_leaf_switch_bw`; aggregate egress per SU is still `compute_rack_count × compute_trays_per_rack × nic_per_tray × nic_speed_gbytes` (see [#scale-out-bandwidth](#scale-out-bandwidth) below).

| Base | L200 twin | Scale-up GPUs | Scale-up BW (GB/s) | Scale-out NIC (GB/s / link, uni-dir) | `scalable_unit_count` |
|------|-----------|---------------|--------------------|--------------------------------------|------------------------|
| b200 | b200l200 | 72 → 1,152 | 900 → 3,600 | 50 → 50 | 4 → 1 |
| b300 | b300l200 | 72 → 1,152 | 900 → 3,600 | 100 → 100 | 4 → 1 |
| r200 | r200l200 | 72 → 1,152 | 1,800 → 7,800 | 100 → 100 | 4 → 1 |
| r576 | r576l200 | 576 → **1,152** | 1,800 → **7,200** (= **4 ×** r200’s **1,800** domain aggregate) | 100 → 100 | 4 → 1 |

**r576l200** uses **`scaleup_switch_port_count=1152`** and **`scaleup_fabric_port_count=72`** at **100** GB/s/link → **72 × 100 = 7,200** GB/s aggregate scale-up (**4 ×** the **r200** base **1,800** GB/s), aligned with the same **1,152**-GPU scale-up domain as the other `*l200` configs.

---

# Memory size (per device)

From `target_config/specs/*.txtpb` `device_memory_size` (bytes → GB).

| Device   | Memory size (GB) |
|----------|------------------|
| b200     | 178              |
| b300     | 288              |
| r200     | 288              |
| r576     | 1,024            |

`*l200` device files match the same chip (e.g. `r576l200` = 1 TiB like `r576`).

---

# Memory bandwidth (per device)

From `target_config/specs/*.txtpb` `memory_bandwidth` (bytes/s → TB/s). Per-device HBM/GDDR bandwidth.

| Device   | Memory bandwidth (TB/s) |
|----------|--------------------------|
| b200     | 7.67                     |
| b300     | 7.94                     |
| r200     | 22                       |
| r576     | 53                       |

---

# Memory capacity (scale-up domain)

Total device memory in one scale-up domain: `scaleup_switch_port_count × device_memory_size` (GPUs in scale-up domain × per-device memory). From configs + specs.

| Config     | Scale-up domain size (GPUs) | Per-device memory (GB) | Domain memory capacity (GB) |
|------------|----------------------------|-------------------------|-----------------------------|
| b200       | 72                         | 178                     | 12,816                      |
| b200l200   | 1,152                      | 178                     | 205,056                     |
| b300       | 72                         | 288                     | 20,736                      |
| b300l200   | 1,152                      | 288                     | 331,776                     |
| r200       | 72                         | 288                     | 20,736                      |
| r200l200   | 1,152                      | 288                     | 331,776                     |
| r576       | 576                        | 1,024                   | 589,824                     |
| r576l200   | 1,152                      | 1,024                   | 1,179,648                   |

---

# Scale-up bandwidth

Aggregate scale-up fabric bandwidth: `scaleup_fabric_port_count × scaleup_fabric_link_bw_gbytes` (GB/s). From configs.

| Config     | Scale-up BW (GB/s) | Scale-up domain size (GPUs) |
|------------|--------------------|-----------------------------|
| b200       | 900                | 72                          |
| b200l200   | 3,600              | 1,152                       |
| b300       | 900                | 72                          |
| b300l200   | 3,600              | 1,152                       |
| r200       | 1,800              | 72                          |
| r200l200   | 7,800              | 1,152                       |
| r576       | 1,800              | 576                         |
| r576l200   | 7,200              | 1,152                       |

### r200 vs r200l200: why 7,800 is not 4 × 1,800

- **r200:** `scaleup_fabric_port_count=18`, `scaleup_fabric_link_bw_gbytes=100` → **18 × 100 = 1,800** GB/s.
- **r200l200:** `scaleup_fabric_port_count=78` (not 72) → **78 × 100 = 7,800** GB/s.

So **7,800 / 1,800 ≈ 4.33×**, not **4×**. This is **not a bug in the arithmetic**; the L200 config uses **78** fabric ports. If the intent were exactly **4 × 1,800 = 7,200** GB/s, you would set **`scaleup_fabric_port_count=72`** at the same link speed (72 × 100).

**r576l200** uses **72 × 100 = 7,200** GB/s (**4 × 1,800**) on purpose, with **1,152** GPUs in the scale-up domain (same as other `*l200` configs).

---

# Scale-up domain size

Number of GPUs in one scale-up domain. From config: `scaleup_switch_port_count` (see `cluster_config.cc`: `scaleup_domain_.scaleup_device_count = scaleup_switch_port_count_`).

| Config     | Scale-up domain size (GPUs) |
|------------|-----------------------------|
| b200       | 72                          |
| b200l200   | 1,152                       |
| b300       | 72                          |
| b300l200   | 1,152                       |
| r200       | 72                          |
| r200l200   | 1,152                       |
| r576       | 576                         |
| r576l200   | 1,152                       |

---

# Scale-out bandwidth

Aggregate NIC bandwidth per scalable unit (egress to scale-out):  
`compute_rack_count × compute_trays_per_rack × nic_per_tray × nic_speed_gbytes` (GB/s). From configs.

Per-link unidirectional rate is **`nic_speed_gbytes`** (50 for **b200** / **b200l200**; 100 for **b300** / **b300l200** / **r200** / **r200l200** / **r576** / **r576l200** in the checked-in configs).

| Config     | Scale-out BW per scalable unit (GB/s) | Scalable units in cluster |
|------------|----------------------------------------|---------------------------|
| b200       | 28,800                                 | 4                         |
| b200l200   | 460,800                                | 1                         |
| b300       | 57,600                                 | 4                         |
| b300l200   | 921,600                                | 1                         |
| r200       | 57,600                                 | 4                         |
| r200l200   | 921,600                                | 1                         |
| r576       | 57,600                                 | 4                         |
| r576l200   | 921,600                                | 1                         |

---

# Scale-out domain size

Number of GPUs in one scalable unit (one scale-out “pod”). From config: `compute_rack_count × compute_trays_per_rack × compute_units_per_tray` (see `cluster_config.cc` `GpuIdToScaleoutCoordinates`: `scalable_unit_size = rack_count × tray_count × unit_count`).

| Config     | Scale-out domain size (GPUs per scalable unit) |
|------------|-------------------------------------------------|
| b200       | 576                                             |
| b200l200   | 9,216                                           |
| b300       | 576                                             |
| b300l200   | 9,216                                           |
| r200       | 576                                             |
| r200l200   | 9,216                                           |
| r576       | 576                                             |
| r576l200   | 9,216                                           |
