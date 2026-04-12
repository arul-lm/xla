# Batch Size Modifier — C FFI

This document describes the C FFI (foreign function interface) for the batch size modifier. Use it when calling the modifier from Rust, Python (ctypes/cffi), or any other language that can call C APIs.

---

## Purpose

The FFI exposes the same logic as the [CLI](#cli-same-logic): read an HLO-variant module (typically saved as `.mlir`), rewrite all batch-dependent tensor dimensions from one batch size to another, and write the modified module to disk. The rewriting covers tensor shapes (`type[dims]`), `slice_sizes={...}` (gather ops), `dynamic_slice_sizes={...}`, constants, slice bounds, and communication operations.

It mirrors the TokenSim `modify_batch_size_v4.py` pipeline: vinveli-style YAML config (`modify_batch_size` section), optional `comm_stats.csv` / `comp_stats.csv` for instruction-level guidance, mesh inference from paths containing `SPxEPxTP`, and an optional reshape consistency pass.

---

## Header and library

| Item | Location |
|------|----------|
| **C header** | `xla/service/gpu/model/batch_size_modifier_c.h` |
| **Implementation** | C entry point in `batch_size_modifier_ffi.cc`; calls `RunBatchSizeModification` in `batch_size_modifier_lib` |

Build the shared bridge (includes analytical latency FFI, peak ops, and batch size modifier):

```bash
bazel build //xla/service/gpu/model:libxla_bridge_all.so
```

The symbol to resolve is **`batch_size_modifier_run`**. All string parameters are **UTF-8** and **null-terminated** unless noted.

---

## Function signature

```c
int batch_size_modifier_run(
    const char* input_mlir_path,
    const char* output_mlir_path,
    int old_batch_size,
    int new_batch_size,
    const char* config_yaml_path,
    const char* path_for_mesh_inference,
    const char* comm_stats_csv_path,
    const char* comp_stats_csv_path,
    int enable_reshape_fix,
    int strict_mode,
    char* error_buffer,
    size_t error_buffer_size);
```

---

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| **input_mlir_path** | `const char*` | Yes | Path to the input HLO module file. Must be non-NULL and non-empty. |
| **output_mlir_path** | `const char*` | Yes | Path for the output file (created/overwritten). Must be non-NULL and non-empty. |
| **old_batch_size** | `int` | Yes | The batch size present in the input module. Must be positive. |
| **new_batch_size** | `int` | Yes | The target batch size. Must be positive. If equal to `old_batch_size`, the input is copied to output unchanged. |
| **config_yaml_path** | `const char*` | Yes | Path to a YAML file containing a `modify_batch_size:` block (see [Config file format](#config-file-format)). Must be non-NULL and non-empty. |
| **path_for_mesh_inference** | `const char*` | No | A file path substring used to find a `NxMxK` mesh shape (SP, EP, tensor/TP) in directory names. NULL or empty defaults to `input_mlir_path`. |
| **comm_stats_csv_path** | `const char*` | No | Path to `comm_stats.csv` with `instruction_name` and `opcode` columns. NULL or empty selects regex-based collective detection. |
| **comp_stats_csv_path** | `const char*` | No | Path to `comp_stats.csv`. Accepts `inst` as alias for `instruction_name` and `group` as alias for `opcode`. NULL or empty is fine. |
| **enable_reshape_fix** | `int` | — | Boolean: **non-zero** enables the reshape product consistency fix-up pass. Recommended default: **1**. |
| **strict_mode** | `int` | — | Boolean: **non-zero** fails if a collective operand/result is not classified for batch rewriting. Recommended default: **0**. |
| **error_buffer** | `char*` | No | Optional buffer for an error message on failure. May be **NULL**. |
| **error_buffer_size** | `size_t` | — | Size of `error_buffer` in bytes. Ignored if `error_buffer` is NULL. If provided, the message is null-terminated and truncated to fit. |

---

## Return value and errors

| Return value | Meaning |
|--------------|---------|
| **0** | Success. The modified module has been written to `output_mlir_path`. |
| **Non-zero** | Failure. If `error_buffer` is non-NULL and `error_buffer_size` > 0, the error message is written there (null-terminated). |

Typical failures:

- Missing or empty required string (`input_mlir_path`, `output_mlir_path`, `config_yaml_path`).
- `old_batch_size` or `new_batch_size` ≤ 0.
- Config YAML missing `modify_batch_size:` section or required fields.
- Input file not found or not readable.
- Output directory does not exist or is not writable.
- `strict_mode` enabled and an unclassified collective is encountered.

---

## Config file format

The YAML config must contain a top-level `modify_batch_size:` section:

```yaml
modify_batch_size:
  num_experts: 256        # Total number of MoE experts (must be > 0; use 1 as placeholder if non-MoE)
  seq_len: 8192           # Sequence length used when generating the input module
  num_experts_per_tok: 8  # Top-k experts per token (must be > 0; use 1 as placeholder if non-MoE)
  sp: 1                   # Sequence parallelism degree (default 1)
  num_heads: 144          # Optional: number of attention heads
```

| Field | Required | Description |
|-------|----------|-------------|
| `num_experts` | Yes | Total expert count. **Must be positive** (`batch_size_modifier_run.cc`); for non-MoE workloads use a dummy such as **1**. |
| `seq_len` | Yes | Sequence length of the input module. Used to detect folded dimensions like `batch * seq_len`. **Must be positive.** |
| `num_experts_per_tok` | Yes | Top-k experts per token. **Must be positive**; for non-MoE use a dummy such as **1**. |
| `sp` | No | Sequence parallelism degree. Default 1 (no sequence parallelism). When `sp > 1`, the modifier detects sharded sequence dimensions (`batch * seq_len / sp`). |
| `num_heads` | No | Number of attention heads. Used to detect `batch * num_heads` folded dimensions. |

The `ep` (expert parallelism) and `tensor_count` (tensor parallelism) values are inferred from the mesh shape in the file path (`SPxEPxTP`), not from the config.

---

## What the modifier rewrites

The modifier performs text-level dimension substitution across the following HLO constructs:

| Construct | Example | What changes |
|-----------|---------|-------------|
| **Tensor shapes** | `f32[4093,8192,128]` → `f32[1,8192,128]` | Dimensions matching `old_batch`, `old_batch * seq_len`, `old_batch * num_experts_per_tok`, etc. |
| **slice_sizes** | `slice_sizes={4093,2,8192,1}` → `slice_sizes={1,2,8192,1}` | Batch-dependent dimensions in gather operations. |
| **dynamic_slice_sizes** | `dynamic_slice_sizes={4093,4}` → `dynamic_slice_sizes={1,4}` | Same for dynamic-slice operations. |
| **Constants** | `s32[] constant(4093)` → `s32[] constant(1)` | Scalar constants equal to the old batch size. |
| **Slice bounds** | `[0:4093]` → `[0:1]` | Slice start/end bounds. |
| **shape/dimensions attrs** | `shape={4093,...}` / `dimensions={4093}` | Attribute values. |
| **Communication ops** | All-reduce, all-gather, reduce-scatter, collective-permute | Operand and result dimensions, guided by `comm_stats.csv` when provided. |

### Structural considerations

The XLA compiler may generate structurally different HLO graphs for different batch sizes (e.g., ring-attention conditionals appear only above a certain batch size threshold for large `seq_len` with `sp > 1`). Since the modifier performs text-level substitution without adding or removing instructions, modifying a reference module with structure A to a batch size whose native compilation produces structure B will result in a latency gap.

When validating the modifier, group batch sizes by MLIR structure (e.g., line count) and use a reference from within each structural class. The validation harness (`validate_batch_size_modification.py`) does this automatically.

---

## Using from another project

### 1. Build or obtain `libxla_bridge_all.so`

```bash
bazel build //xla/service/gpu/model:libxla_bridge_all.so
# Output: bazel-bin/xla/service/gpu/model/libxla_bridge_all.so
```

### 2. Include the header

Copy `xla/service/gpu/model/batch_size_modifier_c.h` into your project, or add the XLA source tree to your include path.

### 3. Load and call

#### Minimal C example (dlopen)

```c
#include <dlfcn.h>
#include <stdio.h>
#include "batch_size_modifier_c.h"

typedef int (*batch_size_modifier_run_fn)(
    const char*, const char*, int, int, const char*, const char*,
    const char*, const char*, int, int, char*, size_t);

int main(void) {
    void* lib = dlopen("./libxla_bridge_all.so", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }

    batch_size_modifier_run_fn run =
        (batch_size_modifier_run_fn)dlsym(lib, "batch_size_modifier_run");

    char err[512];
    int rc = run(
        "/data/hlo/deepseek_r1.mlir",     /* input_mlir_path */
        "/data/hlo/modified.mlir",         /* output_mlir_path */
        4093,                              /* old_batch_size */
        512,                               /* new_batch_size */
        "/data/config/deepseek_r1.yaml",   /* config_yaml_path */
        NULL,                              /* path_for_mesh_inference (use input path) */
        "/data/stats/comm_stats.csv",      /* comm_stats_csv_path */
        "/data/stats/comp_stats.csv",      /* comp_stats_csv_path */
        1,                                 /* enable_reshape_fix = true */
        0,                                 /* strict_mode = false */
        err,
        sizeof(err));

    if (rc != 0) {
        fprintf(stderr, "batch_size_modifier error: %s\n", err);
    }
    dlclose(lib);
    return rc;
}
```

#### Python (ctypes)

```python
import ctypes

lib = ctypes.CDLL("./libxla_bridge_all.so")

lib.batch_size_modifier_run.restype = ctypes.c_int
lib.batch_size_modifier_run.argtypes = [
    ctypes.c_char_p,  # input_mlir_path
    ctypes.c_char_p,  # output_mlir_path
    ctypes.c_int,     # old_batch_size
    ctypes.c_int,     # new_batch_size
    ctypes.c_char_p,  # config_yaml_path
    ctypes.c_char_p,  # path_for_mesh_inference
    ctypes.c_char_p,  # comm_stats_csv_path
    ctypes.c_char_p,  # comp_stats_csv_path
    ctypes.c_int,     # enable_reshape_fix
    ctypes.c_int,     # strict_mode
    ctypes.c_char_p,  # error_buffer
    ctypes.c_size_t,  # error_buffer_size
]

err_buf = ctypes.create_string_buffer(512)
rc = lib.batch_size_modifier_run(
    b"/data/hlo/deepseek_r1.mlir",
    b"/data/hlo/modified.mlir",
    4093, 512,
    b"/data/config/deepseek_r1.yaml",
    None,                                  # use input path for mesh inference
    b"/data/stats/comm_stats.csv",
    b"/data/stats/comp_stats.csv",
    1,                                     # enable_reshape_fix
    0,                                     # strict_mode
    err_buf, 512,
)
if rc != 0:
    raise RuntimeError(f"batch_size_modifier failed: {err_buf.value.decode()}")
```

#### Rust (libloading)

```rust
use libloading::{Library, Symbol};
use std::ffi::CString;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lib = unsafe { Library::new("./libxla_bridge_all.so")? };

    type RunFn = unsafe extern "C" fn(
        *const i8, *const i8, i32, i32, *const i8, *const i8,
        *const i8, *const i8, i32, i32, *mut i8, usize,
    ) -> i32;

    let run: Symbol<RunFn> = unsafe { lib.get(b"batch_size_modifier_run")? };

    let input = CString::new("/data/hlo/deepseek_r1.mlir")?;
    let output = CString::new("/data/hlo/modified.mlir")?;
    let config = CString::new("/data/config/deepseek_r1.yaml")?;
    let comm = CString::new("/data/stats/comm_stats.csv")?;
    let comp = CString::new("/data/stats/comp_stats.csv")?;

    let mut err_buf = vec![0i8; 512];
    let rc = unsafe {
        run(
            input.as_ptr(), output.as_ptr(),
            4093, 512,
            config.as_ptr(),
            std::ptr::null(),  // path_for_mesh_inference
            comm.as_ptr(), comp.as_ptr(),
            1, 0,              // enable_reshape_fix, strict_mode
            err_buf.as_mut_ptr(), err_buf.len(),
        )
    };
    if rc != 0 {
        let msg = unsafe { std::ffi::CStr::from_ptr(err_buf.as_ptr()) };
        eprintln!("Error: {}", msg.to_string_lossy());
    }
    Ok(())
}
```

### C++ in-tree

Prefer `xla::gpu::RunBatchSizeModification` from `batch_size_modifier_core.h` with `BatchSizeModifierOptions` instead of the C entry point. Both CLI and FFI are thin wrappers around this function.

---

## CLI (same logic)

The CLI binary uses the same shared library (`batch_size_modifier_lib`) as the C FFI. Any fix applied to the core library is automatically available to both.

```bash
bazel build //xla/service/gpu/model:batch_size_modifier

./bazel-bin/xla/service/gpu/model/batch_size_modifier \
  --input=/path/to/deepseek_r1.mlir \
  --output=/path/to/out.mlir \
  --old-batch-size=4093 \
  --new-batch-size=512 \
  --config=/path/to/vinveli_config.yaml \
  --comm-stats-csv=/path/to/comm_stats.csv \
  --comp-stats-csv=/path/to/comp_stats.csv
```

Optional flags: `--mesh-inference-path` (defaults to `--input`), `--no-reshape-fix`, `--strict`.

Note: TSL flag parsing requires `--flag=value` format (not `--flag value`).

---

## Validation harness

`xla/tools/batch_size_modifier/validate_batch_size_modification.py` runs the full Vinveli + analytical-latency comparison using the C++ binary. Set **`BATCH_SIZE_MODIFIER_BIN`**, **`--batch-size-modifier-bin`**, or **`XLA_ROOT` / `--xla-root`** so the script can find `bazel-bin/xla/service/gpu/model/batch_size_modifier`.

**One-shot wrapper** (builds the binary, then runs the validator with the right paths):

```bash
VINVELI_HOME=/path/to/vinveli \
  xla/tools/batch_size_modifier/run_validate_batch_size_modification.sh \
    --seq-len 8192 --strategy prefill --max-batch-size 4093 \
    --dtype fp8 --quant q0 --mesh-shape 1x72x1 \
    --config /path/to/model_configs/deepseek_r1.yaml \
    --hardware-arch b200,b200l200 \
    --num-datapoints 10 --max-workers 10
```

The harness automatically groups batch sizes by MLIR structure and selects a reference from within each structural class.

---

## Related

- [Analytical latency calculator C FFI](ANALYTICAL_LATENCY_CALCULATOR_C_FFI.md) (also in `libxla_bridge_all.so`)
