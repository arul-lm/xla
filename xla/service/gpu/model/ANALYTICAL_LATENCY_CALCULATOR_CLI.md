# Analytical Latency Calculator — CLI interface

This document describes the **command-line interface** of the XLA GPU Analytical Latency Calculator. It is intended for use by other projects (e.g. benchmarks, orchestration, or external tooling) that invoke the calculator as a standalone binary.

## Overview

The calculator estimates **compute and communication latency** for an HLO (High-Level Operations) module on one or more GPU cluster configurations. It loads an HLO module from a file, runs cost analysis per instruction and per hardware architecture, applies an optional compute–communication overlap factor, and writes CSV reports to a chosen output directory.

**Binary name (when built from XLA):** `analytical_latency_calculator`  
**Entry point:** `xla/service/gpu/model/analytical_latency_calculator.cc`  
**Core logic:** `xla/service/gpu/model/analytical_latency_calculator_core.cc` (shared with the C FFI library).

---

## Invocation

```text
analytical_latency_calculator \
  --hlo-module-file=<path> \
  --hardware-architectures=<arch1>,<arch2>,... \
  --output-dir=<path> \
  --output-path-prefix=<path> \
  --gpu-model-data-root=<path> \
  --mesh-shape=<dim0>,<dim1>,<dim2> \
  --overlap-factor=<0.0-1.0> \
  [--fix-ragged-dot-flops] \
  [--dump-modified-module]
```

All of `--hlo-module-file`, `--hardware-architectures`, `--output-dir`, `--output-path-prefix`, `--gpu-model-data-root`, `--mesh-shape`, and `--overlap-factor` are **required**. The rest are optional. The binary may also register additional XLA debug flags (see `--help`).

---

## Required flags

| Flag | Description |
|------|-------------|
| `--hlo-module-file` | Path to the HLO module file. Format is inferred from the extension: `.hlo`/`.txt`/`.mlir` = HLO text (`.mlir` means the file content is a HLO variant, not raw MLIR). `.mhlo` = MHLO, `.stablehlo` = StableHLO, `.pb`/`.pbtxt` = protobuf. |
| `--hardware-architectures` | Comma-separated list of hardware architecture **names**. Each name must match a GPU spec file under `gpu_model_data_root` (see [Data layout](#data-layout)). Example: `b200,r200,b200l200`. |
| `--output-dir` | Directory where CSV outputs will be written. Can be relative (then resolved against `--output-path-prefix`) or absolute. |
| `--output-path-prefix` | Base path used when `--output-dir` is relative. Example: repo root `/path/to/xla` or current dir `.`. |
| `--gpu-model-data-root` | Root directory for GPU model data. Under it the tool expects the standard XLA layout: `xla/backends/gpu/target_config/specs/` (device `.txtpb` files) and `xla/service/gpu/model/configs/` (cluster `.config` files). |
| `--mesh-shape` | 3D mesh shape as three comma-separated positive integers (e.g. `4,4,4`). Used for communication cost and partitioning. |
| `--overlap-factor` | Compute–communication overlap factor in the range **0.0** to **1.0**. `0.0` = no overlap (total time = compute + communication). `1.0` = full overlap. |

---

## Optional flags

| Flag | Description |
|------|-------------|
| `--fix-ragged-dot-flops` | Enable a fix for ragged_dot FLOP calculation (replaces certain dot instructions with ragged-dot variants for cost analysis). |
| `--scale-memory-bandwidth` | Scale the device memory bandwidth by this factor (default **1.0**). Use e.g. **10** to simulate higher bandwidth; if total time drops a lot, the workload is memory-bound. See [Verify memory-bound with scaled bandwidth](#verify-memory-bound-with-scaled-bandwidth). |
| `--dump-modified-module` | If set, write the (possibly modified) HLO module to `modified_module.hlo` in the output directory. |
| `--num-pipeline-stages` | **Calcium-only.** Number of pipeline stages. Default **`1`** = no PP modeling (output is byte-identical to runs without this flag). Values **`> 1`** are rejected for any non-Calcium architecture (`r200`, `b200`, `b300`, `tpu`, `r576`, `rcpx`, …). See [Pipeline parallelism (Calcium-only)](#pipeline-parallelism-calcium-only). |
| `--pipeline-activation-bytes` | **Calcium-only.** Per-stage activation bytes for the inter-stage handoff. Default **`0`** auto-infers from the HLO entry-computation parameter shapes. For training (forward + backward) pass `2 * forward_size` explicitly. |
| `--pipeline-microbatches` | **Calcium-only.** Number of microbatches in the pipeline step. Default **`0`** reports per-microbatch metrics only (`t_total_us = 0`); use a positive value to compute total runtime and bubble fraction. |
| `--pipeline-comm-overlap-factor` | **Calcium-only (Step 8b).** Inter-stage compute/handoff overlap factor in `[0.0, 1.0]`. Default **`0.0`** = no overlap (byte-stable). The visible per-boundary handoff cost is multiplied by `(1 - factor)`; see [Pipeline parallelism (Calcium-only)](#pipeline-parallelism-calcium-only). |

**How `--scale-memory-bandwidth` is applied in code:** For each hardware architecture, the tool loads the device spec (e.g. `r200.txtpb`) and reads `memory_bandwidth` from `GpuDeviceInfoProto`. If the scale is not 1.0, it overwrites that value with `memory_bandwidth * scale_memory_bandwidth` before building the `DeviceDescription`. All subsequent read/write time estimates in the performance model (`ReadTimeWithDRAMHeuristic`, `WriteTime`) use this scaled bandwidth, so a higher scale shortens memory time and reduces total latency when the workload is memory-bound. The same option is available in the C FFI as `scale_memory_bandwidth` (see `analytical_latency_calculator_c.h`).

---

## Data layout under `--gpu-model-data-root`

The tool resolves paths relative to `gpu_model_data_root` as follows:

- **Specs (device descriptions):**  
  `{gpu_model_data_root}/xla/backends/gpu/target_config/specs/{arch}.txtpb`  
  Each entry in `--hardware-architectures` must have a corresponding `{arch}.txtpb` (e.g. `b200` → `b200.txtpb`).

- **Cluster configs:**  
  `{gpu_model_data_root}/xla/service/gpu/model/configs/{arch}.config`  
  The same `{arch}` name is used to load the cluster config (e.g. `b200` → `b200.config`).

**Example:** If `gpu_model_data_root=/path/to/xla`, then for `--hardware-architectures=b200,r200` the tool will look for:

- `/path/to/xla/xla/backends/gpu/target_config/specs/b200.txtpb`
- `/path/to/xla/xla/backends/gpu/target_config/specs/r200.txtpb`
- `/path/to/xla/xla/service/gpu/model/configs/b200.config`
- `/path/to/xla/xla/service/gpu/model/configs/r200.config`

Typical architecture names shipped in XLA include: `b200`, `b200l200`, `b300`, `b300l200`, `r200`, `r200l200`, `rcpx`, `rcpxl200`, and others present under the specs/configs directories.

The Calcium architecture (`q250`) is also supported (see [Pipeline parallelism (Calcium-only)](#pipeline-parallelism-calcium-only)). Its spec (`q250.txtpb`) and cluster config (`q250.config`) are deployed locally and are not part of the public-checked-in set; provide them under the standard paths above when running with `--hardware-architectures=q250`.

---

## Output directory and files

The **effective output path** is:

- If `--output-dir` is **absolute:** that path is used as-is.
- If `--output-dir` is **relative:** it is joined with `--output-path-prefix` (e.g. `output_path_prefix/output_dir`).

The tool creates the output directory if it does not exist, then writes the following CSV files (and optionally one HLO file) into it:

| File | Description |
|------|-------------|
| `device_stats.csv` | Per-device (per-architecture) summary: overlapped/original latency, compute/comm time, overlap factor and savings. |
| `comp_stats.csv` | Per-computation and per-instruction compute stats: latency, **tflops** (FLOPs used for compute_time, in units of 10¹²), bytes read/written, compute_time, read_time, write_time, throughput_tflops_per_sec, **effective_matrix_tflops** (hardware TFLOPS actually applied to the op's FLOPs to compute compute_time, when on matrix-op path), etc. |
| `comm_stats.csv` | Communication cost stats: instruction name, opcode, comm type, cost (µs), replica group size, bandwidth/volume fields. |
| `overlap_stats.csv` | Overlap summary per device: overlap factor, original/overlapped total time, compute/comm time, savings, percentage. |
| `instruction_timeline.csv` | Per-instruction timeline: device, instruction name, type (compute/comm/other), duration (µs), start/end time. |
| `compute_op_counts.csv` | Per-architecture counts of compute ops by datatype (e.g. f4e2m1fn, f8e4m3fn). |
| `compute_memory_bound_stats.csv` | Per-architecture counts: **compute_bound** (ops where compute_time ≥ read_time + write_time) and **memory_bound** (ops where read+write &gt; compute_time). |
| `pipeline_stats.csv` | Written only when `--num-pipeline-stages > 1` is used with at least one Calcium architecture. Contains per-stage and pipeline-fill metrics. See [Pipeline parallelism (Calcium-only)](#pipeline-parallelism-calcium-only). |
| `modified_module.hlo` | Written only if `--dump-modified-module` is set; contains the HLO module after any in-memory modifications (e.g. ragged-dot fix). |

All CSVs use comma separators and include a header row. Data is **appended** across multiple hardware architectures (one block per architecture), so a single run with `--hardware-architectures=b200,r200` produces one set of files with rows for both b200 and r200.

**FLOPs used for compute_time:** The **tflops** column in **comp_stats.csv** is the FLOP count that is actually used to compute **compute_time** for that instruction, in units of 10¹² (i.e. raw FLOPs = tflops × 10¹²). It comes from the cost analysis (`flop_count`) and is passed unchanged into the performance model (e.g. `ComputeTimeFromPeakMatrixOps(device_info_, flops, instr)` for dots). So no separate column is needed; **tflops** already tracks that input.

**Effective hardware matrix TFLOPS:** The **effective_matrix_tflops** column is the **effective** hardware matrix TFLOPS actually used to compute **compute_time** for that op: i.e. the rate such that compute_time = (op FLOPs) / (this rate in FLOP/s). It is derived as (tflops × 10¹²) / compute_time_sec when the instruction is on the matrix-op path (e.g. dots on Rubin/Blackwell). Empty when the op did not use the matrix-unit path.

---

## Exit codes and errors

- **0** — Success.
- **1** — Failure: missing required flag, invalid value (e.g. bad `--overlap-factor` or `--mesh-shape`), missing spec/config file, or internal error. A short message is printed to stdout.

Validation rules enforced by the CLI:

- `--overlap-factor` must parse as a number in **[0.0, 1.0]**.
- `--mesh-shape` must be exactly **three** positive integers (e.g. `4,4,4`); extra dimensions or non-numeric values cause an error.
- At least one non-empty hardware architecture must be given; each must have a corresponding `{arch}.txtpb` under the specs dir.

---

## Example (other projects)

Assume another project has:

- An HLO module at `./my_module.hlo`
- A copy or symlink of XLA’s GPU model data at `./xla_gpu_model` (so that `xla_gpu_model/xla/backends/gpu/target_config/specs/` and `xla_gpu_model/xla/service/gpu/model/configs/` exist).

Then:

```bash
analytical_latency_calculator \
  --hlo-module-file=./my_module.hlo \
  --hardware-architectures=b200,r200 \
  --output-dir=./results/run1 \
  --output-path-prefix=. \
  --gpu-model-data-root=./xla_gpu_model \
  --mesh-shape=4,4,4 \
  --overlap-factor=0.5
```

Output CSVs will be under `./results/run1/`. To use an absolute output path:

```bash
analytical_latency_calculator \
  --hlo-module-file=/data/models/my_module.hlo \
  --hardware-architectures=b200 \
  --output-dir=/data/results/run1 \
  --output-path-prefix=/data/results \
  --gpu-model-data-root=/path/to/xla \
  --mesh-shape=2,4,8 \
  --overlap-factor=0.0
```

---

## Library usage (C++ / C )

The same core logic is available as a library so other projects can call it without spawning the CLI:

- **C++:** Use `RunAnalyticalLatencyCalculation()` from `xla/service/gpu/model/analytical_latency_calculator_core.h` with a populated `AnalyticalLatencyCalculatorOpts` and a 3-element `mesh_shape`. No defaults; all options must be set by the caller.
- **C:** Use the C FFI in `analytical_latency_calculator_c.h` / `analytical_latency_calculator_ffi.cc` to pass paths and options and run the same pipeline.

### C FFI: module file and format

The C library receives **`hlo_module_file`**: a path to the module file. Examples:

- `.../hlo/hlo_pre_deepseek_r1_q0_fp8_64_1048576_1x72x1/deepseek_r1.mlir`
- `.../hlo/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir`

**Format is inferred from the file extension.** The extension is `.mlir` but the **file content is a HLO variant** (HLO text), not raw MLIR. The loader therefore treats `.mlir` as HLO text:

| Extension    | Inferred format | Notes |
|-------------|------------------|--------|
| `.hlo`, `.txt`, `.mlir` | HLO text | `.mlir` = file contains HLO variant, not MLIR. |
| `.mhlo`     | MHLO             | Translated to HLO (for actual MLIR/MHLO content). |
| `.stablehlo`| StableHLO        | Translated to HLO. |
| `.pb`, `.pbtxt` | Protobuf     | HLO snapshot. |

So for the paths above, both are loaded as HLO; FP4 vs FP8 (and other dtypes) in the HLO content are preserved and produce different compute times in the results.

### Why might FP4 vs FP8 show only a small difference in compute time?

**Root cause (fixed):** The model used the dot instruction’s **result** element type for peak matrix ops. Many graphs use the same **accumulation** type (e.g. F32 or BF16) for both FP4 and FP8 dots, so both runs used the same peak TFLOPS and gave very similar compute time. The fix: for **dot** and **ragged_dot**, peak matrix ops now use the **first operand’s** element type (compute precision), so FP4 vs FP8 dots get different peak TFLOPS and different compute times.

Other possible reasons:

1. **Same dtype in the HLO**  
   If both files were exported with the same element type (e.g. both BF16 or both FP8), the model will use the same peak matrix ops for dot instructions, so compute time will be almost identical. The path name (fp4 vs fp8) does not affect the result; only the **element types in the HLO** do.

2. **Matmul FLOPs are the same**  
   FLOPs for a dot are `2*M*N*K` regardless of FP4 vs FP8. The only difference is **peak TFLOPS** (FP4 higher on r200). So if dtypes differ, you should still see a noticeable ratio (e.g. FP4 ~30–40% lower compute time for dot-bound workloads). A “very small” difference suggests either the same dtype or that most time is not in dot ops.

3. **Memory- or comm-bound**  
   Total instruction latency is a blend of compute, read, and write time. If most time is read/write or communication, changing peak matrix ops has limited effect. Check `comp_stats.csv`: compare `compute_time` vs `read_time` and `write_time` for the heavy ops.

4. **Most cost is not from dots**  
   If most latency comes from non-dot ops (e.g. collectives, elementwise), their cost is dtype-independent or uses a different path, so FP4 vs FP8 will look similar.

### Verify FP4 vs FP8 with the CLI

Run the calculator twice (once per file), then compare `comp_stats.csv` and `device_stats.csv`. Use one architecture to keep output small.

```bash
# Set these to your actual paths
HLO_ROOT="/path/to/your/hlo"
GPU_MODEL_ROOT="/path/to/xla"   # or your gpu_model_data_root
OUT_PREFIX="/tmp/alc"

# FP8 run
analytical_latency_calculator \
  --hlo-module-file="${HLO_ROOT}/hlo_pre_deepseek_r1_q0_fp8_64_1048576_1x72x1/deepseek_r1.mlir" \
  --hardware-architectures=r200 \
  --output-dir=fp8_run \
  --output-path-prefix="${OUT_PREFIX}" \
  --gpu-model-data-root="${GPU_MODEL_ROOT}" \
  --mesh-shape=1,72,1 \
  --overlap-factor=0.5

# FP4 run
analytical_latency_calculator \
  --hlo-module-file="${HLO_ROOT}/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir" \
  --hardware-architectures=r200 \
  --output-dir=fp4_run \
  --output-path-prefix="${OUT_PREFIX}" \
  --gpu-model-data-root="${GPU_MODEL_ROOT}" \
  --mesh-shape=1,72,1 \
  --overlap-factor=0.5
```

Then inspect:

- **`comp_stats.csv`**  
  - **`datatype`** column: FP4 dots should show `f4e2m1fn`, FP8 dots `f8e4m3fn` (or similar). If both show the same datatype, the HLO content has the same element types and that explains the similar compute time.  
  - **`latency(µs)`**, **`compute_time`**: compare the same `op_name` / `inst` across the two runs; dot ops should have lower compute time for FP4 if dtypes differ.

- **`device_stats.csv`**  
  - Compare `compute_time_secs` (and `overlapped_latency_secs`) between `fp8_run` and `fp4_run`.

Example (after running):

```bash
# Compare datatype and latency for a few dot-like rows
grep -E "dot|matmul|einsum" "${OUT_PREFIX}/fp8_run/comp_stats.csv" | head -20
grep -E "dot|matmul|einsum" "${OUT_PREFIX}/fp4_run/comp_stats.csv" | head -20

# Compare total compute time per device
cat "${OUT_PREFIX}/fp8_run/device_stats.csv"
cat "${OUT_PREFIX}/fp4_run/device_stats.csv"
```

If `datatype` is identical in both CSVs for the matrix ops, the two `.mlir` files contain the same element types; the difference in compute time will stay small until the HLO is exported with distinct types (F4E2M1FN vs F8E4M3FN).

### Run FP4 and FP8 and save stats in separate directories

Use **different `--output-dir`** for each run so FP4 and FP8 stats go to separate directories (and are not appended into the same CSVs).

**Docker (bazel run):**

```bash
# FP8 → /xla/fp8_stats/
docker exec xla bazel run --spawn_strategy=sandboxed //xla/service/gpu/model:analytical_latency_calculator -- \
  --hlo-module-file=/xla/hlo/hlo_pre_deepseek_r1_q0_fp8_64_1048576_1x72x1/deepseek_r1.mlir \
  --hardware-architectures=r200 \
  --output-dir=fp8_stats \
  --output-path-prefix=/xla \
  --gpu-model-data-root=/xla \
  --mesh-shape=1,72,1 \
  --overlap-factor=1.0 \
  --fix-ragged-dot-flops

# FP4 → /xla/fp4_stats/
docker exec xla bazel run --spawn_strategy=sandboxed //xla/service/gpu/model:analytical_latency_calculator -- \
  --hlo-module-file=/xla/hlo/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir \
  --hardware-architectures=r200 \
  --output-dir=fp4_stats \
  --output-path-prefix=/xla \
  --gpu-model-data-root=/xla \
  --mesh-shape=1,72,1 \
  --overlap-factor=1.0 \
  --fix-ragged-dot-flops
```

Output: **FP8** stats in `/xla/fp8_stats/`, **FP4** stats in `/xla/fp4_stats/` (paths inside the container; under your mount on the host). Compare with:

```bash
docker exec xla cat /xla/fp8_stats/device_stats.csv
docker exec xla cat /xla/fp4_stats/device_stats.csv
docker exec xla grep -E "dot|matmul|einsum" /xla/fp8_stats/comp_stats.csv | head -20
docker exec xla grep -E "dot|matmul|einsum" /xla/fp4_stats/comp_stats.csv | head -20
```

After the dot-operand fix, FP4 runs should show lower compute time for dot ops when the HLO uses F4E2M1FN on the dot operands.

### Why is the FP4 performance improvement so small?

From a typical run (DeepSeek R1 FP8 vs FP4 HLO, r200, mesh 1×72×1):

| Metric | FP8 | FP4 | Difference |
|--------|-----|-----|------------|
| **compute_time_secs** | 1205.33 | 1197.11 | **−8.22 s (~0.7%)** |
| **overlapped_latency_secs** | 1280.81 | 1273.08 | −7.73 s |
| **comm_time_secs** | 81.23 | 81.23 | 0 (unchanged) |

So FP4 improves total compute only slightly. Reasons:

1. **Most of the “FP4” HLO is still FP8 (or shared dtype).**  
   The model uses the **first operand’s** element type for dot peak TFLOPS. In the FP4 run, only a **subset** of dot instructions actually have **F4E2M1FN** on the first operand (e.g. MoE ragged-dots and some others). The rest (e.g. many attention/embedding matmuls) still use **F8E4M3FN** or the same type as in the FP8 run, so their compute time is unchanged. So the “FP4” module is mixed-precision: only the ops that are truly FP4 get the higher peak (50 TFLOPS sparse on r200) and lower compute time.

2. **Per-op speedup is large where FP4 is used.**  
   Where the HLO does use FP4 operands, the calculator reflects it: e.g. the same attention dot goes from **1.265 s** (FP8) to **0.900 s** (FP4), and ragged-dots from **8.65 ms** to **6.16 ms** per op. So the **model and peak TFLOPS are correct**; the small overall gain is because most of the 1205 s of compute is from dots that still use the same (e.g. FP8) operand type in both runs.

3. **To see a larger FP4 gain** you need an HLO where **most** dot operands are F4E2M1FN (e.g. a fully FP4-quantized model). Then total compute would drop roughly by the peak ratio (e.g. ~1.4× for r200 FP4 vs FP8 sparse).

**When compute_op_counts shows mostly FP4 (e.g. f4e2m1fn=1026) but the time difference is still low:** The cost that is summed into total time is **exec_time** per instruction, not raw compute_time. The model uses `exec_time = compute_time + memory_time - 0.95×min(compute_time, memory_time)`. When **memory dominates** (read+write &gt; compute), exec_time ≈ memory_time + 0.05×compute_time, so most of the op duration is memory (same for FP4 and FP8). Cutting compute_time by 40% then only changes the 5% part, so total time barely drops. Check **comp_stats.csv**: if read_time + write_time &gt; compute_time for most dot rows, the workload is memory-bound.

### Verify memory-bound with scaled bandwidth

You can **temporarily increase memory bandwidth** in the model with **`--scale-memory-bandwidth`**. If the workload is memory-bound, total time should drop significantly when bandwidth is scaled up; if it is compute-bound, total time will change little.

**Example (Docker):**

```bash
# Baseline (r200 nominal bandwidth)
docker exec xla bazel run --spawn_strategy=sandboxed //xla/service/gpu/model:analytical_latency_calculator -- \
  --hlo-module-file=/xla/hlo/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir \
  --hardware-architectures=r200 --output-dir=temp_stats --output-path-prefix=/xla \
  --gpu-model-data-root=/xla --mesh-shape=1,72,1 --overlap-factor=1.0 --fix-ragged-dot-flops

# 10× memory bandwidth (simulated)
docker exec xla bazel run --spawn_strategy=sandboxed //xla/service/gpu/model:analytical_latency_calculator -- \
  --hlo-module-file=/xla/hlo/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir \
  --hardware-architectures=r200 --output-dir=temp_stats_10xbw --output-path-prefix=/xla \
  --gpu-model-data-root=/xla --mesh-shape=1,72,1 --overlap-factor=1.0 --fix-ragged-dot-flops \
  --scale-memory-bandwidth=10
```

Compare **overlapped_latency_secs** (or **compute_time_secs**) in `device_stats.csv` between the two runs. A large drop with `--scale-memory-bandwidth=10` supports the theory that most ops are memory-bound.

The CLI is a thin wrapper that parses flags into `AnalyticalLatencyCalculatorOpts` and `mesh_shape`, then calls `RunAnalyticalLatencyCalculation()`.

---

## Pipeline parallelism (Calcium-only)

Calcium (`q250`) is the only architecture that participates in pipeline-parallelism modeling in this calculator. All other architectures (`tpu`, `b200`, `b300`, `r200`, `r576`, `rcpx`, …) assume **`PP=1`**: the HLO module is treated as the whole workload. To model a multi-stage pipeline on Calcium, pass the **HLO of one pipeline stage** together with three new optional flags:

| Flag | Meaning |
|------|---------|
| `--num-pipeline-stages=S` | Number of pipeline stages. `1` (default) = no PP modeling and output is byte-identical to runs without the flag. `> 1` is **only valid for Calcium**; non-Calcium archs get an explicit error. |
| `--pipeline-activation-bytes=B` | Per-stage activation byte size for the inter-stage handoff. `0` (default) auto-infers from the HLO entry-computation parameter shapes (sum of `ShapeUtil::ByteSizeOf` over all parameters). For training (forward + backward) pass `2 * forward_size` explicitly. |
| `--pipeline-microbatches=M` | Number of microbatches in the pipeline step. `0` (default) reports per-microbatch metrics only (`t_total_us = 0`). `> 0` computes total runtime and bubble fraction. |
| `--pipeline-comm-overlap-factor=A` | **(Step 8b)** Inter-stage compute/handoff overlap factor in `[0.0, 1.0]`. `0.0` (default) = no overlap modeled (byte-stable with pre-Step-8b runs). `0.5` = typical 1F1B-style overlap with the next stage's compute. The visible per-boundary handoff cost is multiplied by `(1 - A)`. |

### Bubble accounting (forward-only inference)

The handoff is now computed **per stage boundary**: for each `k ∈ [0, S-2]` the calculator walks the Calcium fabric path between rank-0 of stage `k` and rank-0 of stage `k+1`, and exposes both a **raw** view (no overlap applied) and an **effective** view (raw × `(1 - pipeline_comm_overlap_factor)`):

- **raw** is the cost on the critical path of any single microbatch — overlap cannot hide it because a microbatch can't enter stage `k+1` until its own handoff completes.
- **effective** is what the next microbatch sees in steady state, since the handoff overlaps with the next microbatch's compute on the previous stage.

```text
T_first       = S * T_stage + t_handoff_sum (raw)        (mb 0 traversal)
T_step        = max(T_stage, t_handoff_max (effective))  (steady-state cadence)
T_total(M)    = T_first + (M - 1) * T_step               (M >= 1; 0 when M = 0)
bubble_time   = (S - 1) * T_stage + t_handoff_sum (raw)
bubble_frac   = bubble_time / T_total
```

For `pipeline_comm_overlap_factor = 0.0` the raw and effective views are identical and these formulas reduce to the legacy `T_first = S*T_stage + (S-1)*T_handoff` (assuming uniform per-boundary cost), which is byte-stable with pre-Step-8b runs.

The per-boundary aggregates come from `CalciumClusterConfig::CalculatePipelineHandoffCosts`. For each boundary it routes a 1-to-1 P2P transfer through the Calcium fabric (L1 / L2 / L3 PCIe + L4 RoCE v2 for q250; optical interposer / optical switch / L4 for q250l200), takes the bottleneck `EffectiveBandwidth(...)` with `sharing=1`, applies the `internode_efficiency_factor`, and adds 1-µs-per-hop latency. `devices_per_stage` is `product(mesh_shape)`.

The `bound_by` column in `pipeline_stats.csv` compares `T_stage` against `t_handoff_max_us` (effective; drives steady-state cadence): `compute` if `T_stage` dominates, `comm` if `t_handoff_max` dominates, and `balanced` if they are within ~10% of each other.

### TTFT for streamed microbatches

When `M` microbatches of a single global batch are submitted at `t = 0` and stream through the pipeline, the first-output (TTFT) of the sequences in microbatch `m` is:

```text
TTFT_m = T_first + m * T_step                  for m in [0, M-1]
```

so:

| Metric | Formula |
|---|---|
| TTFT_min (first microbatch) | `T_first` |
| TTFT_max (last microbatch) | `T_first + (M-1) * T_step` = `T_total(M)` |
| TTFT_mean | `T_first + ((M-1)/2) * T_step` |
| TTFT_p95 | `T_first + 0.95 * (M-1) * T_step` |

Increasing `M` improves throughput (smaller `bubble_fraction` because the pipe is amortized over more microbatches) but increases `TTFT_max` linearly because the last microbatch waits for all `M-1` prior microbatches at the steady-state cadence.

### Example: 4 stages, 8 microbatches

```bash
analytical_latency_calculator \
  --hlo-module-file=/tmp/one_stage.hlo \
  --hardware-architectures=q250 \
  --output-dir=/tmp/pp_run \
  --output-path-prefix=. \
  --gpu-model-data-root=/path/to/xla \
  --mesh-shape=1,8,1 \
  --overlap-factor=0.0 \
  --num-pipeline-stages=4 \
  --pipeline-activation-bytes=4194304 \
  --pipeline-microbatches=8
```

Inspect `/tmp/pp_run/pipeline_stats.csv`:

```text
device_name,hw_arch,num_pipeline_stages,pipeline_microbatches,pipeline_activation_bytes,devices_per_stage,t_stage_us,t_handoff_us,t_first_us,t_step_us,t_total_us,bubble_time_us,bubble_fraction,bound_by,t_handoff_max_us,t_handoff_sum_us,pipeline_comm_overlap_factor
Calcium SoC q250,q250,4,8,4194304,8,7197.83,77.10,29022.62,7197.83,79407.42,21824.79,0.27,compute,77.10,231.30,0.0
```

The last three columns are Step 8b additions (always present; `0.0` for `pipeline_comm_overlap_factor` means the v1-default no-overlap behavior).

> **Two overlap factors, distinct knobs.** `--overlap-factor` is the **within-stage** compute/communication overlap that shapes `t_stage_us` (and everything downstream of it). `--pipeline-comm-overlap-factor` is the **inter-stage** handoff/compute overlap that shapes only `t_step_us` (steady-state cadence). The CSV column `pipeline_comm_overlap_factor` echoes only the second knob; the first one is baked into `t_stage_us` and surfaced in `overlap_stats.csv`. They compose multiplicatively — see the C FFI doc's "Two distinct overlap factors" section.

### Non-Calcium architectures

Running PP > 1 with any other architecture is rejected at the validation layer:

```bash
$ analytical_latency_calculator --hardware-architectures=r200 --num-pipeline-stages=4 ...
Error: Pipeline parallelism (num_pipeline_stages=4) is only supported for q250/calcium.
       Got: r200. Other architectures assume PP=1; pass num_pipeline_stages=1.
exit=1
```

The same reject applies in the C FFI — see [ANALYTICAL_LATENCY_CALCULATOR_C_FFI.md](ANALYTICAL_LATENCY_CALCULATOR_C_FFI.md).

### Backward compatibility

- Omitting all three PP flags, or passing `--num-pipeline-stages=1`, produces CSV output that is **byte-identical** to runs without these flags (validated with `diff -r`).
- `pipeline_stats.csv` is created **only** when `num_pipeline_stages > 1` and at least one Calcium architecture is processed.

---

## Docker / bazel run: arguments and output location

**Arguments:** The calculator only defines the flags listed in [Required flags](#required-flags) and [Optional flags](#optional-flags). Flags such as `--format`, `--all`, `--gpu`, `--print-collective-details` come from XLA debug options (`AppendDebugOptionsFlags`) and are accepted but **not used** for module loading or format; format is always inferred from the file extension. You can omit them. Your required/optional set is valid:

- `--hlo-module-file`, `--hardware-architectures=r200`, `--output-dir=temp_stats`, `--output-path-prefix=al_stats`, `--gpu-model-data-root="/xla"`, `--mesh-shape=1,72,1`, `--overlap-factor=1.0`, `--fix-ragged-dot-flops` are correct.

**Where the output stats files are:** The effective output path is `output_path_prefix` + `output_dir`, i.e. **`al_stats/temp_stats`**, relative to the **current working directory** of the process.

With `docker exec xla bazel run ...`, the cwd is usually the workspace root inside the container (e.g. the directory that contains the XLA repo). So the CSV files are at:

```text
<workspace_root>/al_stats/temp_stats/device_stats.csv
<workspace_root>/al_stats/temp_stats/comp_stats.csv
<workspace_root>/al_stats/temp_stats/comm_stats.csv
<workspace_root>/al_stats/temp_stats/overlap_stats.csv
<workspace_root>/al_stats/temp_stats/instruction_timeline.csv
<workspace_root>/al_stats/temp_stats/compute_op_counts.csv
<workspace_root>/al_stats/temp_stats/compute_memory_bound_stats.csv
<workspace_root>/al_stats/temp_stats/pipeline_stats.csv  # only with --num-pipeline-stages>1 on Calcium
```

**Compute op counts by datatype:** The calculator writes **`compute_op_counts.csv`** with columns `hw_arch`, `datatype`, `count`. Each row is the number of compute instructions (dots, ragged-dots, and other ops with non-zero cost) that use that datatype for **compute/peak** (for dot/ragged_dot this is the first operand’s element type). It also prints a one-line summary per architecture, e.g. `Compute op counts (r200): f4e2m1fn=174, f8e4m3fn=852, ...`. Use this to see how many ops use FP4, FP8, or BF16. The **`comp_stats.csv`** `datatype` column uses the same convention (compute precision for dots).

**Compute-bound vs memory-bound:** The calculator counts ops where **compute_time ≥ read_time + write_time** as **compute-bound**, and the rest as **memory-bound**. It prints e.g. `Compute-bound ops (r200): 100, Memory-bound ops (r200): 926` and writes **`compute_memory_bound_stats.csv`** with columns `hw_arch`, `compute_bound`, `memory_bound`.

To find them:

1. **Inside the container** (workspace is often `/xla` or similar):
   ```bash
   docker exec xla ls -la /xla/al_stats/temp_stats/
   # or, if cwd was elsewhere when you ran:
   docker exec xla find / -name "device_stats.csv" 2>/dev/null
   ```
2. **On the host:** If the repo is mounted (e.g. `-v /home/arul/dev/xla:/xla`), then the same path under your mount is the host path, e.g. `/home/arul/dev/xla/al_stats/temp_stats/`.

To force output to a path that’s easy to find on the host, use an absolute `--output-dir` so `--output-path-prefix` is ignored:

```bash
docker exec xla bazel run --spawn_strategy=sandboxed //xla/service/gpu/model:analytical_latency_calculator -- \
  --hlo-module-file=/xla/hlo/hlo_pre_deepseek_r1_q0_fp4_64_1048576_1x72x1/deepseek_r1.mlir \
  --hardware-architectures=r200 \
  --output-dir=/xla/al_stats/temp_stats \
  --output-path-prefix=/xla \
  --gpu-model-data-root=/xla \
  --mesh-shape=1,72,1 \
  --overlap-factor=1.0 \
  --fix-ragged-dot-flops
```

Then the CSVs are at **`/xla/al_stats/temp_stats/`** inside the container (and under your mount on the host).
