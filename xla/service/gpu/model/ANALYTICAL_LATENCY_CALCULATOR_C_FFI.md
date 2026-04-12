# Analytical Latency Calculator — C FFI

This document describes the C FFI (foreign function interface) for the analytical latency calculator. Use it when calling the calculator from Rust, Python (ctypes/cffi), or any other language that can call C APIs.

---

## Purpose

The FFI exposes the same logic as the [CLI](ANALYTICAL_LATENCY_CALCULATOR_CLI.md): load an HLO module, run cost and communication analysis for one or more GPU architectures, and write CSV outputs (device stats, compute stats, comm stats, overlap stats, instruction timeline, etc.). There are no optional flags beyond what is passed as function arguments.

---

## Header and library

| Item | Location |
|------|----------|
| **C header** | `xla/service/gpu/model/analytical_latency_calculator_c.h` |
| **Implementation** | `analytical_latency_calculator_ffi.cc` (target `analytical_latency_calculator_ffi`) |

The **`analytical_latency_calculator`** CLI binary links **`analytical_latency_calculator_core`** and `analytical_latency_calculator.cc` only; it does **not** link the FFI translation unit. The C symbol **`analytical_latency_calculator_run`** is exported from shared libraries that depend on `analytical_latency_calculator_ffi`, e.g. **`//xla/service/gpu/model:libanalytical_latency_ffi.so`** or **`//xla/service/gpu/model:libxla_bridge_all.so`** (see `BUILD`).

All string parameters must be **UTF-8** and **null-terminated**.

---

## Function signature

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

---

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| **hlo_module_file** | `const char*` | Yes | Path to the HLO module file. Format is inferred from extension: `.hlo` / `.txt` = HLO text; `.mlir` = HLO text (content is HLO variant, not raw MLIR); `.mhlo` / `.stablehlo` = MLIR. Must be non-NULL and non-empty. |
| **hardware_architectures** | `const char*` | Yes | Comma-separated list of architecture names (e.g. `"r200,b200"`). Each must have a matching spec under `gpu_model_data_root` (e.g. `r200.txtpb`). At least one architecture required. |
| **output_dir** | `const char*` | Yes | Directory where CSV files are written. Can be absolute or relative to the current working directory. Created if it does not exist. |
| **gpu_model_data_root** | `const char*` | Yes | Root for model data. Specs are loaded from `{gpu_model_data_root}/xla/backends/gpu/target_config/specs/{arch}.txtpb`, cluster configs from `{gpu_model_data_root}/xla/service/gpu/model/configs/{arch}.config`. |
| **mesh_shape** | `const char*` | Yes | Exactly three positive integers, comma-separated (e.g. `"1,72,1"` or `"4,4,4"`). No spaces required; leading/trailing whitespace is trimmed per dimension. |
| **overlap_factor** | `double` | Yes | Compute–communication overlap factor in **[0.0, 1.0]**. Same meaning as CLI `--overlap-factor`. |
| **fix_ragged_dot_flops** | `int` | Yes | Boolean: **0** = false, **non-zero** = true. Same as CLI `--fix-ragged-dot-flops`. |
| **dump_modified_module** | `int` | Yes | Boolean: **0** = false, **non-zero** = true. If true, writes the (possibly modified) HLO module to `modified_module.hlo` in `output_dir`. Same as CLI `--dump-modified-module`. |
| **scale_memory_bandwidth** | `double` | Yes | Scale factor for device memory bandwidth. **1.0** = no change. Must be **> 0**. Use e.g. **10.0** to simulate 10× bandwidth (useful to check if a workload is memory-bound). See [How scale_memory_bandwidth is applied](#how-scale_memory_bandwidth-is-applied) below. |
| **error_buffer** | `char*` | No | Optional buffer for an error message on failure. May be **NULL**. |
| **error_buffer_size** | `size_t` | — | Size of `error_buffer` in bytes. Ignored if `error_buffer` is NULL. If provided, the message is null-terminated and truncated to fit. |

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
- Parse or internal errors; message is placed in `error_buffer`.

---

## Validation rules (enforced by the FFI)

- **hlo_module_file**, **output_dir**, **gpu_model_data_root**, **mesh_shape**: must be non-NULL and have length > 0.
- **hardware_architectures**: after splitting on commas and trimming, at least one non-empty architecture name must remain.
- **scale_memory_bandwidth**: must be greater than 0.
- **mesh_shape**: must parse as exactly three positive integers (e.g. `"1,72,1"`). Negative or zero values, non-numeric tokens, or a different number of dimensions cause an error.

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

Details and column descriptions are in [ANALYTICAL_LATENCY_CALCULATOR_CLI.md](ANALYTICAL_LATENCY_CALCULATOR_CLI.md) (Output directory and files, and the following sections).

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

## Minimal C example

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

---

## Relation to the CLI and other FFIs

- **CLI:** The [analytical latency calculator CLI](ANALYTICAL_LATENCY_CALCULATOR_CLI.md) parses flags (e.g. `--scale-memory-bandwidth`, `--mesh-shape`) into `AnalyticalLatencyCalculatorOpts` and the same `mesh_shape` vector, then calls `RunAnalyticalLatencyCalculation()`. The C FFI does the same call with arguments supplied by the host language.
- **Peak matrix ops FFI:** A separate C API, **`peak_matrix_ops_per_ns_from_hw_arch`** (see `peak_matrix_ops_shard_c.h`), returns peak matrix ops per nanosecond for a given hardware spec (text proto) and dtype. That value is **sparse** (2× the dense spec). It does not run the full latency calculation; use `analytical_latency_calculator_run` for that.
