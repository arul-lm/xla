# Analytical Latency Calculator — C FFI

This document describes the C FFI (foreign function interface) for the analytical latency calculator. Use it when calling the calculator from Rust, Python (ctypes/cffi), or any other language that can call C APIs.

---

## Purpose

The FFI exposes the same logic as the [CLI](ANALYTICAL_LATENCY_CALCULATOR_CLI.md): load an HLO module, run cost and communication analysis for one or more GPU architectures, and write CSV outputs (device stats, compute stats, comm stats, overlap stats, instruction timeline, etc.). There are no optional flags beyond what is passed as function arguments.

Two C entry points are exported:

| Function | Use it when |
|----------|-------------|
| **`analytical_latency_calculator_run`** | You want the legacy behavior (no pipeline parallelism). All architectures are supported and assume `PP=1`. **ABI-stable**: signature has not changed since this function was introduced. Forwards internally to `analytical_latency_calculator_run_with_pipeline` with `num_pipeline_stages = 1` and `pipeline_comm_overlap_factor = 0.0`. |
| **`analytical_latency_calculator_run_with_pipeline`** | You want pipeline-parallelism modeling for the Calcium (`q250`) architecture, with bubble accounting and a `pipeline_stats.csv` output. Includes a `pipeline_comm_overlap_factor` knob in `[0.0, 1.0]` for inter-stage compute/handoff overlap (`0.0` = no overlap, byte-stable with pre-Step-8b runs). Setting `num_pipeline_stages <= 1` makes this function byte-identical to `analytical_latency_calculator_run`. |

Both symbols are exported from the FFI shared libraries (verify with `nm -D libanalytical_latency_ffi.so | grep analytical_latency`). The `_with_pipeline` signature gained `pipeline_comm_overlap_factor` in Step 8b — pre-existing callers of that symbol must recompile against the new header. The legacy `_run` symbol is unchanged.

---

## Header and library

| Item | Location |
|------|----------|
| **C header** | `xla/service/gpu/model/analytical_latency_calculator_c.h` |
| **Implementation** | `analytical_latency_calculator_ffi.cc` (target `analytical_latency_calculator_ffi`) |

The **`analytical_latency_calculator`** CLI binary links **`analytical_latency_calculator_core`** and `analytical_latency_calculator.cc` only; it does **not** link the FFI translation unit. The C symbols **`analytical_latency_calculator_run`** and **`analytical_latency_calculator_run_with_pipeline`** are exported from shared libraries that depend on `analytical_latency_calculator_ffi`, e.g. **`//xla/service/gpu/model:libanalytical_latency_ffi.so`** or **`//xla/service/gpu/model:libxla_bridge_all.so`** (see `BUILD`).

All string parameters must be **UTF-8** and **null-terminated**.

---

## Function signatures

### Legacy entry point (ABI-stable)

```c
int analytical_latency_calculator_run(
    const char* hlo_module_file,
    const char* hardware_architectures,
    const char* output_dir,
    const char* gpu_model_data_root,
    const char* mesh_shape,
    double overlap_factor,
    int fix_ragged_dot_flops,
    int dump_modified_module,
    double scale_memory_bandwidth,
    char* error_buffer,
    size_t error_buffer_size);
```

This function is internally implemented as a thin forwarder to `analytical_latency_calculator_run_with_pipeline` with all pipeline-parallelism fields zeroed. Pre-built downstream binaries that link against this symbol continue to work unchanged.

### Pipeline-parallelism entry point (Calcium-only)

```c
int analytical_latency_calculator_run_with_pipeline(
    const char* hlo_module_file,
    const char* hardware_architectures,
    const char* output_dir,
    const char* gpu_model_data_root,
    const char* mesh_shape,
    double overlap_factor,
    int fix_ragged_dot_flops,
    int dump_modified_module,
    double scale_memory_bandwidth,
    int     num_pipeline_stages,           /* 0 or 1 = no PP modeling */
    int64_t pipeline_activation_bytes,     /* 0 = auto-infer from HLO */
    int     pipeline_microbatches,         /* 0 = per-microbatch only */
    double  pipeline_comm_overlap_factor,  /* 0.0-1.0, 0.0 = no overlap */
    char* error_buffer,
    size_t error_buffer_size);
```

The first nine parameters and the trailing `error_buffer` / `error_buffer_size` are identical to the legacy entry point. The four middle parameters control pipeline-parallelism modeling. `pipeline_comm_overlap_factor` was added in Step 8b — pre-existing callers must recompile against the new header. Setting `num_pipeline_stages <= 1` (and any value for the others) makes this function byte-identical to `analytical_latency_calculator_run`.

---

## Parameters

### Common parameters (both entry points)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| **hlo_module_file** | `const char*` | Yes | Path to the HLO module file. Format is inferred from extension: `.hlo` / `.txt` = HLO text; `.mlir` = HLO text (content is HLO variant, not raw MLIR); `.mhlo` / `.stablehlo` = MLIR. Must be non-NULL and non-empty. |
| **hardware_architectures** | `const char*` | Yes | Comma-separated list of architecture names (e.g. `"r200,b200"`). Each must have a matching spec under `gpu_model_data_root` (e.g. `r200.txtpb`). At least one architecture required. |
| **output_dir** | `const char*` | Yes | Directory where CSV files are written. Can be absolute or relative to the current working directory. Created if it does not exist. |
| **gpu_model_data_root** | `const char*` | Yes | Root for model data. Specs are loaded from `{gpu_model_data_root}/xla/backends/gpu/target_config/specs/{arch}.txtpb`, cluster configs from `{gpu_model_data_root}/xla/service/gpu/model/configs/{arch}.config`. |
| **mesh_shape** | `const char*` | Yes | Exactly three positive integers, comma-separated (e.g. `"1,72,1"` or `"4,4,4"`). No spaces required; leading/trailing whitespace is trimmed per dimension. |
| **overlap_factor** | `double` | Yes | **Within-stage** compute–communication overlap factor in **[0.0, 1.0]**. Controls how much within-stage comm hides behind within-stage compute when computing `t_stage_us` (one microbatch on one stage). Same meaning as CLI `--overlap-factor`. **Different from `pipeline_comm_overlap_factor`** — see [Two distinct overlap factors](#two-distinct-overlap-factors). |
| **fix_ragged_dot_flops** | `int` | Yes | Boolean: **0** = false, **non-zero** = true. Same as CLI `--fix-ragged-dot-flops`. |
| **dump_modified_module** | `int` | Yes | Boolean: **0** = false, **non-zero** = true. If true, writes the (possibly modified) HLO module to `modified_module.hlo` in `output_dir`. Same as CLI `--dump-modified-module`. |
| **scale_memory_bandwidth** | `double` | Yes | Scale factor for device memory bandwidth. **1.0** = no change. Must be **> 0**. Use e.g. **10.0** to simulate 10× bandwidth (useful to check if a workload is memory-bound). See [How scale_memory_bandwidth is applied](#how-scale_memory_bandwidth-is-applied) below. |
| **error_buffer** | `char*` | No | Optional buffer for an error message on failure. May be **NULL**. |
| **error_buffer_size** | `size_t` | — | Size of `error_buffer` in bytes. Ignored if `error_buffer` is NULL. If provided, the message is null-terminated and truncated to fit. |

### Pipeline-parallelism parameters (`_with_pipeline` only)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| **num_pipeline_stages** | `int` | Yes (use `1` for default) | Number of pipeline stages. **`0` or `1`** disables PP modeling and outputs are byte-identical to `analytical_latency_calculator_run`. **`> 1`** enables PP modeling and is **only valid for Calcium (`q250`)**; other architectures return an error. |
| **pipeline_activation_bytes** | `int64_t` | Yes (use `0` for default) | Byte size of the activation handed off between consecutive pipeline stages. **`0`** auto-infers from the HLO entry-computation parameter shapes (sum of `xla::ShapeUtil::ByteSizeOf` over all parameters). For training (forward + backward) pass `2 * forward_size` explicitly. |
| **pipeline_microbatches** | `int` | Yes (use `0` for default) | Number of microbatches per pipeline step. **`0`** reports per-microbatch metrics only (`t_total_us = 0` in `pipeline_stats.csv`); **`> 0`** computes the full pipeline runtime including bubble fraction. |
| **pipeline_comm_overlap_factor** | `double` | Yes (use `0.0` for default) | Inter-stage compute/handoff overlap factor in `[0.0, 1.0]`. The visible per-boundary handoff cost is multiplied by `(1 - factor)`. `0.0` = no overlap, conservative (and byte-stable with pre-Step-8b runs). `0.5` ≈ typical 1F1B-style overlap with the next stage's compute. `1.0` = handoff fully hidden in cadence. Out-of-range values are clamped. |

The HLO module passed to `_with_pipeline` is interpreted as **one pipeline stage**. The calculator multiplies by `num_pipeline_stages` and computes a **per-boundary** handoff cost via `CalciumClusterConfig::CalculatePipelineHandoffCosts`: for each `k ∈ [0, S-2]` it walks `PathBetweenDevices(k * devices_per_stage, (k+1) * devices_per_stage)`, takes the bottleneck `EffectiveBandwidth(...)` with `sharing=1`, applies the `internode_efficiency_factor`, adds 1-µs-per-hop latency, and exposes two views:

- **raw**: the unmodified per-boundary handoff cost (no overlap applied). This is the cost on the critical path of any single microbatch, since a microbatch cannot enter stage `k+1` until its own handoff `k → k+1` finishes.
- **effective**: raw × `(1 - pipeline_comm_overlap_factor)`. This models how much of each handoff is hidden behind the **next** microbatch's compute on stage `k` in steady state.

Pipeline-level metrics use these views differently — overlap helps cadence but cannot help any single microbatch's traversal cost:

```
T_first    = num_stages * T_stage  +  t_handoff_raw_sum    (mb 0 traversal)
T_step     = max(T_stage, t_handoff_max)                   (steady-state cadence)
T_total(M) = T_first + (M - 1) * T_step                    for M >= 1, else 0
bubble     = (num_stages - 1) * T_stage  +  t_handoff_raw_sum
```

The exposed columns:

| Column | View | Meaning |
|--------|------|---------|
| `t_handoff_max_us` | effective | Slowest effective boundary; drives `T_step`. |
| `t_handoff_sum_us` | **raw** | Sum of raw per-boundary costs; drives `T_first` and `bubble_time`. (Note: this is the **raw** sum, not the effective sum, because `T_first` cannot be hidden by overlap.) |
| `t_handoff_us` | effective | Average effective per-boundary handoff; diagnostic only. |
| `pipeline_comm_overlap_factor` | — | The factor actually applied. |

For `pipeline_comm_overlap_factor = 0.0` the raw and effective views are identical and the metrics collapse to the legacy `T_first = S*T_stage + (S-1)*T_handoff` form (assuming uniform per-boundary cost), which preserves byte-stable v1 output.

`bound_by` compares `T_stage` against `t_handoff_max` (the cadence-relevant **effective** value) — `compute` / `comm` / `balanced` with a 10% threshold.

### Two distinct overlap factors

The PP path has **two independent overlap knobs** that operate at different layers of the model. Both are scalars in `[0.0, 1.0]` and both default to `0.0`, but they mean very different things:

| Knob | Layer | What it overlaps | Affects |
|---|---|---|---|
| **`overlap_factor`** | Within a single stage | The HLO's per-instruction **compute** with the HLO's per-instruction **comm** (e.g. AllReduce). Computed by `CalculateInstructionLevelOverlap`. | `t_stage_us` (per-microbatch-per-stage cost). Through that, indirectly affects everything downstream. |
| **`pipeline_comm_overlap_factor`** | Between pipeline stages | The **inter-stage activation handoff** with the **next microbatch's compute on the previous stage** in steady state. Applied as `(1 − factor)` to the per-boundary raw handoff. | `t_step_us` only. `t_first_us`, `bubble_time_us`, and any single microbatch's traversal cost use the **raw** handoff regardless of this factor — microbatch 0 in particular cannot hide its own handoffs because it has no predecessor in the pipe. |

They compose multiplicatively: `overlap_factor` first shapes `t_stage_us`; `pipeline_comm_overlap_factor` then shapes `t_step_us` from `t_stage_us` and the per-boundary handoffs. A run with `overlap_factor = 0.7, pipeline_comm_overlap_factor = 0.5` is modelling "70% of within-stage comm hides behind within-stage compute, AND 50% of each inter-stage handoff hides behind the next microbatch's compute".

The CSV column **`pipeline_comm_overlap_factor`** records only the second knob. The first knob is implicitly baked into `t_stage_us` and is not echoed in `pipeline_stats.csv` (it appears in `overlap_stats.csv`).

---

## Return value and errors

| Return value | Meaning |
|--------------|---------|
| **0** | Success. CSVs (and optionally `modified_module.hlo`) have been written to `output_dir`. |
| **Non-zero** | Failure. If `error_buffer` is non-NULL and `error_buffer_size` > 0, the error message is written there (null-terminated). |

Typical failures:

- Missing or empty required string (e.g. `hlo_module_file`, `output_dir`, `gpu_model_data_root`, `mesh_shape`, or no architectures in `hardware_architectures`).
- **scale_memory_bandwidth** ≤ 0.
- **mesh_shape** not exactly three positive integers.
- File not found (module file, spec `.txtpb`, or cluster `.config`).
- **`num_pipeline_stages > 1`** combined with a non-Calcium architecture (e.g. `r200`, `b200`, `b300`, `tpu`, `r576`, `rcpx`). The error message identifies the offending architecture.
- Parse or internal errors; message is placed in `error_buffer`.

---

## Validation rules (enforced by the FFI)

- **hlo_module_file**, **output_dir**, **gpu_model_data_root**, **mesh_shape**: must be non-NULL and have length > 0.
- **hardware_architectures**: after splitting on commas and trimming, at least one non-empty architecture name must remain.
- **scale_memory_bandwidth**: must be greater than 0.
- **mesh_shape**: must parse as exactly three positive integers (e.g. `"1,72,1"`). Negative or zero values, non-numeric tokens, or a different number of dimensions cause an error.
- **Pipeline parallelism** (`_with_pipeline` only): `num_pipeline_stages > 1` is rejected unless **every** architecture in `hardware_architectures` is `q250` / `calcium`. Other architectures assume `PP=1` (the HLO represents the whole workload), so calling `_with_pipeline` with `num_pipeline_stages > 1` for them is treated as a programming error rather than silently ignored. Use `analytical_latency_calculator_run` (or pass `num_pipeline_stages = 1`) for those archs. The reject is also enforced inside `RunAnalyticalLatencyCalculation`, so the CLI behaves the same way.

The FFI does **not** validate **overlap_factor** beyond passing it through. The core’s overlap routine uses **`CHECK`** that **overlap_factor** is in **[0.0, 1.0]** (`CalculateInstructionLevelOverlap` in `analytical_latency_calculator_core.cc`); values outside that range **abort the process** rather than returning a message in **error_buffer**. Spec/config existence is checked via **`ValidateHardwareArchitectures`** and returns **`absl::Status`** to the FFI (surfaced in **error_buffer**).

---

## Output files

The same CSV and HLO files as the CLI are produced under `output_dir`:

- `device_stats.csv`
- `comp_stats.csv`
- `comm_stats.csv`
- `overlap_stats.csv`
- `instruction_timeline.csv`
- `compute_op_counts.csv`
- `compute_memory_bound_stats.csv`
- `modified_module.hlo` (only if `dump_modified_module` is non-zero)
- **`pipeline_stats.csv`** (only if `_with_pipeline` was called with `num_pipeline_stages > 1` **and** at least one Calcium architecture was processed)

`pipeline_stats.csv` columns (Step 8b adds the last three; the existing columns are unchanged in name and order, so v1 CSV consumers keep working):

| Column | Meaning |
|--------|---------|
| `device_name` | Device name from the spec, e.g. `Calcium SoC q250`. |
| `hw_arch` | Architecture token, e.g. `q250`. |
| `num_pipeline_stages` | Echo of the input. |
| `pipeline_microbatches` | Echo of the input. |
| `pipeline_activation_bytes` | The value used (auto-inferred when input was 0). |
| `devices_per_stage` | `product(mesh_shape)`. |
| `t_stage_us` | Per-microbatch stage latency = `overlap_stats.original_total_time_us` for the HLO. |
| `t_handoff_us` | **Average effective** per-boundary handoff cost (Step 8b: was the single rank-0→rank-D handoff). For `pipeline_comm_overlap_factor = 0` and uniform per-boundary cost, identical to v1. |
| `t_first_us` | Pipeline fill latency = `S * t_stage + t_handoff_sum_us` (using **raw** sum — overlap cannot hide microbatch 0's traversal). |
| `t_step_us` | Steady-state per-microbatch latency = `max(t_stage, t_handoff_max_us)`. |
| `t_total_us` | Total runtime for `M` microbatches = `t_first_us + (M-1) * t_step_us`; `0` when `pipeline_microbatches == 0`. |
| `bubble_time_us` | Pipeline-fill bubble time = `(S-1) * t_stage + t_handoff_sum_us`. |
| `bubble_fraction` | `bubble_time_us / t_total_us`; `0` when `t_total_us == 0`. |
| `bound_by` | `compute` (stage dominates), `comm` (handoff dominates), or `balanced` (within 10% ratio). Compares `t_stage` against `t_handoff_max_us` (effective). |
| `t_handoff_max_us` | **(Step 8b)** Slowest of the `S-1` **effective** per-boundary handoffs; drives `t_step_us`. |
| `t_handoff_sum_us` | **(Step 8b)** Sum of all `S-1` **raw** per-boundary handoffs; drives `t_first_us` and `bubble_time_us`. (Raw on purpose — see overlap semantics above.) |
| `pipeline_comm_overlap_factor` | **(Step 8b)** The `pipeline_comm_overlap_factor` actually applied (clamped to `[0,1]`). Distinct from the `overlap_factor` parameter, which controls within-stage compute/comm overlap and shapes `t_stage_us`; this column controls inter-stage handoff/compute overlap and shapes `t_step_us` only. |

Details and column descriptions of the other CSVs are in [ANALYTICAL_LATENCY_CALCULATOR_CLI.md](ANALYTICAL_LATENCY_CALCULATOR_CLI.md) (Output directory and files, and the following sections).

---

## How scale_memory_bandwidth is applied

For each hardware architecture, the loader reads the device spec (e.g. `r200.txtpb`) and gets **memory_bandwidth** from the spec’s `GpuDeviceInfoProto`. If **scale_memory_bandwidth** is not **1.0**, it replaces that value with:

```text
memory_bandwidth = memory_bandwidth * scale_memory_bandwidth
```

before building the `DeviceDescription`. All subsequent read/write time estimates in the performance model (`ReadTimeWithDRAMHeuristic`, `WriteTime`) use this scaled bandwidth. So:

- **scale_memory_bandwidth > 1** → higher effective bandwidth → shorter memory time → lower total latency when the workload is memory-bound.
- **scale_memory_bandwidth < 1** → lower effective bandwidth → longer memory time.

Use a value like **10.0** to quickly check whether total time drops a lot when memory is “faster”; if it does, the run is likely memory-bound.

---

## Minimal C example (legacy entry point, no PP)

```c
#include "analytical_latency_calculator_c.h"
#include <stdio.h>

int main(void) {
  char err[512];
  int ret = analytical_latency_calculator_run(
      "/path/to/deepseek_r1.mlir",   /* hlo_module_file */
      "r200",                         /* hardware_architectures */
      "/tmp/out",                     /* output_dir */
      "/path/to/xla",                 /* gpu_model_data_root */
      "1,72,1",                       /* mesh_shape */
      1.0,                            /* overlap_factor */
      1,                              /* fix_ragged_dot_flops = true */
      0,                              /* dump_modified_module = false */
      1.0,                            /* scale_memory_bandwidth = no scaling */
      err,
      sizeof(err));
  if (ret != 0) {
    fprintf(stderr, "Error: %s\n", err);
    return 1;
  }
  return 0;
}
```

## Pipeline-parallelism C example (Calcium / q250)

```c
#include "analytical_latency_calculator_c.h"
#include <stdint.h>
#include <stdio.h>

int main(void) {
  char err[512];
  int ret = analytical_latency_calculator_run_with_pipeline(
      "/path/to/one_stage.hlo",      /* hlo_module_file: ONE pipeline stage */
      "q250",                         /* Calcium-only when num_pipeline_stages > 1 */
      "/tmp/out_pp",                  /* output_dir */
      "/path/to/xla",                 /* gpu_model_data_root */
      "1,8,1",                        /* mesh_shape -> devices_per_stage = 8 */
      0.0,                            /* overlap_factor */
      0,                              /* fix_ragged_dot_flops */
      0,                              /* dump_modified_module */
      1.0,                            /* scale_memory_bandwidth */
      4,                              /* num_pipeline_stages */
      (int64_t)4 * 1024 * 1024,       /* pipeline_activation_bytes (4 MiB) */
      8,                              /* pipeline_microbatches */
      err,
      sizeof(err));
  if (ret != 0) {
    fprintf(stderr, "Error: %s\n", err);
    return 1;
  }
  /* Inspect /tmp/out_pp/pipeline_stats.csv for bubble metrics. */
  return 0;
}
```

Pass `num_pipeline_stages = 1` (and any values for the other two PP fields) to get exactly the legacy behavior — no `pipeline_stats.csv` is written, all other CSVs are byte-identical.

## Forwarder semantics (legacy → with-pipeline)

The body of `analytical_latency_calculator_run` is now a 1-line call:

```c
return analytical_latency_calculator_run_with_pipeline(
    hlo_module_file, hardware_architectures, output_dir,
    gpu_model_data_root, mesh_shape, overlap_factor,
    fix_ragged_dot_flops, dump_modified_module, scale_memory_bandwidth,
    /* num_pipeline_stages */    1,
    /* pipeline_activation_bytes */ 0,
    /* pipeline_microbatches */  0,
    error_buffer, error_buffer_size);
```

Because the legacy entry point pins all three PP fields to their no-op defaults, every CSV produced by `analytical_latency_calculator_run` is byte-identical to the pre-Step-8 calculator (validated by `diff -r`). Existing FFI consumers do not need to recompile or change their call sites.

---

## Relation to the CLI and other FFIs

- **CLI:** The [analytical latency calculator CLI](ANALYTICAL_LATENCY_CALCULATOR_CLI.md) parses flags (including `--num-pipeline-stages`, `--pipeline-activation-bytes`, `--pipeline-microbatches`) into `AnalyticalLatencyCalculatorOpts` and the same `mesh_shape` vector, then calls `RunAnalyticalLatencyCalculation()`. The C FFI does the same call with arguments supplied by the host language. The non-Calcium PP reject is enforced inside `RunAnalyticalLatencyCalculation`, so both layers behave identically.
- **Peak matrix ops FFI:** A separate C API, **`peak_matrix_ops_per_ns_from_hw_arch`** (see `peak_matrix_ops_shard_c.h`), takes the hardware **textproto** (`GpuHloRunnerTargetConfig` / arch spec) and a dtype, and returns a signed peak throughput in **matrix ops per nanosecond**.

  **Compute-unit contract:** For every architecture, the scalar return value is defined **per smallest schedulable compute unit** in that spec—the unit XLA uses when attributing peak matmul throughput from the hardware model. On **traditional GPU** targets that unit is typically **one GPU / one device**. On **Calcium (`q250`)** and similar multi-SoC devices, that unit is **one SoC** (not per card, per server, or pooled across the mesh). Hosts (including Tunnel) must **not** divide or multiply this FFI result to “convert” to another basis; if a UI needs cluster-level totals, it must combine using **explicit device / SoC counts** from the workload or mesh, not by reinterpreting the FFI scalar.

  The value is **sparse** (2× the dense spec) where applicable. It does not run the full latency calculation; use `analytical_latency_calculator_run` (or `_with_pipeline`) for that.

  **Tunnel:** `get_peak_matrix_ops` forwards the `*.txtpb` bytes and dtype to this FFI and displays the returned ops/ns (and TFLOPs derived only by fixed unit conversion ×10⁹ / ×10¹²). It does **not** normalize hardware config bandwidth fields to per-SoC for display; those `hardware_metrics` numbers remain **as authored** in `*.config` (raw link or per-card fields as in the file).
