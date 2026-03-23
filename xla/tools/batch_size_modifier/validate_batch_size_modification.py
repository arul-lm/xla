#!/usr/bin/env python3
"""
Validation script to compare latency between original MLIR files and modified MLIR files.

MLIR FILE PATHS:
===============
1. Original MLIR files are created by run_sdy_generator_batch.py in:
   - VINVELI_HOME/hlo/hlo_{pre|dec}_deepseek_r1_{quant}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}/
   - Example: /data/home/arul/dev/vinveli/hlo/hlo_dec_deepseek_r1_q0_fp8_1_8192_2x72x1/deepseek_r1.mlir
   - NOTE: There is NO option to change this output path - run_sdy_generator_batch.py always saves to VINVELI_HOME/hlo/

2. This script searches for original MLIR files in:
   - VINVELI_HOME/hlo/ (same location where run_sdy_generator_batch.py saves them)

3. Modified MLIR files are saved to:
   - --output-dir/modified_mlir/ (or temp directory if --output-dir not specified)

4. run_analytical_latency_batch.py finds MLIR files by:
   - Reading the CSV file (sdy_gen.csv) with batch_size, seq_len, strategy, mesh_shape, dtype, quant
   - Constructing the path: VINVELI_HOME/hlo/hlo_{pre|dec}_deepseek_r1_{quant}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}/deepseek_r1.mlir
   - NOTE: run_analytical_latency_batch.py also expects files in VINVELI_HOME/hlo/ - there is no option to use an alternate path

This script:
1. (Optional) Clears existing MLIR files for the batch size datapoints under test
2. Generates original MLIR files using run_sdy_generator_batch.py
3. Uses the XLA C++ `batch_size_modifier` tool (see //xla/service/gpu/model:batch_size_modifier)
   to generate modified MLIR files (--batch-size-modifier-bin, BATCH_SIZE_MODIFIER_BIN, or
   bazel-bin under --xla-root / XLA_ROOT)
4. Copies MLIR files to XLA container and runs run_analytical_latency_batch.py for both sets
5. Calculates and prints the latency gap

How run_analytical_latency_batch.py results are collected:
- run_analytical_latency_batch.py reads the CSV file and processes each entry
- For each MLIR file, it runs the analytical latency calculator in the XLA container
- Results are written to device_stats.csv files in subdirectories matching the MLIR file names
- The script copies results from container to local stats directory (specified by --stats-dir)
- Results are organized in subdirectories: stats_dir/{hlo_dir_name}/device_stats.csv

How results are used with modify batch size script:
- The modify batch size script does NOT use analytical latency results directly
- It uses communication stats CSV files (comm_stats.csv) if provided via --comm-stats-csv
- These CSV files help identify communication operations that need batch size updates
- The script can also use compute stats (comp_stats.csv) via --comp-stats-csv
- If CSV files are not provided, the modify batch size script analyzes the MLIR file directly
- The batch size modifier binary is resolved via --batch-size-modifier-bin, BATCH_SIZE_MODIFIER_BIN,
  or `<xla-root>/bazel-bin/xla/service/gpu/model/batch_size_modifier`

How results from modify batch size script MLIR files are processed:
- Modified MLIR files are copied to XLA container (same structure as originals)
- run_analytical_latency_batch.py is run again with a CSV for modified files
- Results are extracted from device_stats.csv files using extract_latency_from_stats()
- The function recursively searches for device_stats.csv in the stats directory
- Latency values are read from 'overlapped_latency_secs' column and converted to milliseconds
- Results are compared between original and modified files to calculate gaps

Usage:
    python3 validate_batch_size_modification.py --seq-len <seq_len> --strategy <strategy> --max-batch-size <max_batch_size> --dtype <dtype> --quant <quant> --mesh-shape <mesh_shape> --hardware-arch <arch>
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate batch size modification by comparing latency between original and modified MLIR files'
    )
    parser.add_argument('--seq-len', type=int, required=True,
                       help='Sequence length')
    parser.add_argument('--strategy', type=str, required=True, choices=['prefill', 'decode'],
                       help='Strategy: prefill or decode')
    parser.add_argument('--max-batch-size', type=int, required=True,
                       help='Maximum batch size (used as old_batch_size for modification)')
    parser.add_argument('--dtype', type=str, required=True,
                       help='Data type (e.g., fp8, bf16)')
    parser.add_argument('--quant', type=str, required=True,
                       help='Quantization (e.g., q0, q1)')
    parser.add_argument('--mesh-shape', type=str, required=True,
                       help='Mesh shape (e.g., 1x48x1)')
    parser.add_argument('--num-datapoints', type=int, default=10,
                       help='Number of datapoints to generate in range [1, max_batch_size]. Default: 10')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config YAML file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for modified MLIR files. Default: temp directory')
    parser.add_argument('--vinveli-home', type=str, default=None,
                       help='VINVELI_HOME path. If not provided, will use VINVELI_HOME environment variable')
    parser.add_argument('--hardware-arch', type=str, required=True,
                       help='Hardware architecture(s) for latency calculation (e.g., b200,b200l200)')
    parser.add_argument('--overlap-factor', type=float, default=0.5,
                       help='Overlap factor for latency calculation. Default: 0.5')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of workers for parallel execution. Default: 1')
    parser.add_argument('--skip-cleanup', action='store_true',
                       help='Skip cleaning MLIR files before generating new ones')
    parser.add_argument('--xla-container-path', type=str, default="/xla/hlo",
                       help='Path in XLA container for MLIR files. Default: /xla/hlo')
    parser.add_argument('--container-name', type=str, default="xla",
                       help='Docker container name. Default: xla')
    parser.add_argument('--batch-size-modifier-bin', type=str, default=None,
                       help='Path to batch_size_modifier (built with '
                            '//xla/service/gpu/model:batch_size_modifier). '
                            'Overrides BATCH_SIZE_MODIFIER_BIN.')
    parser.add_argument('--xla-root', type=str, default=None,
                       help='OpenXLA repo root; used to find bazel-bin/.../batch_size_modifier '
                            'if no binary is specified. Default: XLA_ROOT env.')
    
    return parser.parse_args()


def get_vinveli_home(args) -> Path:
    """Get VINVELI_HOME path."""
    if args.vinveli_home:
        vinveli_home = Path(args.vinveli_home)
    else:
        vinveli_home = os.getenv('VINVELI_HOME')
        if not vinveli_home:
            raise ValueError("VINVELI_HOME environment variable not set. Use --vinveli-home or set VINVELI_HOME")
        vinveli_home = Path(vinveli_home)
    
    if not vinveli_home.exists():
        raise ValueError(f"VINVELI_HOME path does not exist: {vinveli_home}")
    
    return vinveli_home


def get_model_config_path(args) -> Path:
    """Get model config path."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        raise ValueError(f"Model config file does not exist: {config_path}")
    
    return config_path


def generate_batch_sizes(max_batch_size: int, num_files: int = 10) -> List[int]:
    """Generate batch sizes in range [1, max_batch_size]."""
    if max_batch_size < num_files:
        # If max_batch_size is less than num_files, use all values from 1 to max_batch_size
        return list(range(1, max_batch_size + 1))
    
    # Generate evenly spaced batch sizes
    step = max(1, (max_batch_size - 1) // (num_files - 1))
    batch_sizes = [1]
    for i in range(1, num_files - 1):
        batch_size = 1 + i * step
        if batch_size <= max_batch_size:
            batch_sizes.append(batch_size)
    batch_sizes.append(max_batch_size)
    
    # Remove duplicates and sort
    batch_sizes = sorted(list(set(batch_sizes)))
    return batch_sizes


def clear_mlir_files(vinveli_home: Path, batch_sizes: List[int], seq_len: int, 
                    strategy: str, mesh_shape: str, dtype: str, quant: str):
    """Clear existing MLIR files for the batch sizes under test.
    
    This function removes MLIR files in VINVELI_HOME/hlo before generating new ones
    to ensure clean test results.
    """
    print(f"\n{'='*60}")
    print("Step 0: Clearing existing MLIR files for batch sizes under test")
    print(f"{'='*60}")
    print(f"Clearing MLIR files in: {vinveli_home / 'hlo'}")
    
    hlo_path = vinveli_home / "hlo"
    quant_str = quant if quant.startswith('q') else f"q{quant}"
    prefix = "hlo_pre" if strategy == "prefill" else "hlo_dec"
    
    cleared_count = 0
    for batch_size in batch_sizes:
        # Try pattern with dtype and quant first
        hlo_dir_name = f"{prefix}_deepseek_r1_{quant_str}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}"
        hlo_dir = hlo_path / hlo_dir_name
        
        # Try exact match first
        if hlo_dir.exists():
            print(f"  Removing: {hlo_dir.name}")
            shutil.rmtree(hlo_dir)
            cleared_count += 1
            continue
        
        # Try glob pattern
        matching_dirs = list(hlo_path.glob(f"{hlo_dir_name}*"))
        if matching_dirs:
            for match_dir in matching_dirs:
                print(f"  Removing: {match_dir.name}")
                shutil.rmtree(match_dir)
                cleared_count += 1
        
        # Also try pattern without dtype/quant
        hlo_dir_name_simple = f"{prefix}_deepseek_r1_{batch_size}_{seq_len}_{mesh_shape}"
        hlo_dir_simple = hlo_path / hlo_dir_name_simple
        if hlo_dir_simple.exists():
            print(f"  Removing: {hlo_dir_simple.name}")
            shutil.rmtree(hlo_dir_simple)
            cleared_count += 1
        else:
            matching_dirs = list(hlo_path.glob(f"{hlo_dir_name_simple}*"))
            for match_dir in matching_dirs:
                print(f"  Removing: {match_dir.name}")
                shutil.rmtree(match_dir)
                cleared_count += 1
    
    print(f"✓ Cleared {cleared_count} MLIR directories")


def create_sdy_gen_csv(batch_sizes: List[int], seq_len: int, strategy: str, 
                       mesh_shape: str, dtype: str, quant: str, output_path: Path):
    """Create sdy_gen.csv file for run_sdy_generator_batch.py.
    
    Args:
        quant: Can be "q0", "q1", "True", "False", "true", "false", "1", "0", etc.
               Will be converted to boolean format expected by run_sdy_generator_batch.py
    """
    # Convert quant to boolean format expected by run_sdy_generator_batch.py
    # It expects "True" or "False" (or "true"/"false"), not "q0"/"q1"
    if quant.startswith('q'):
        # Convert "q0" -> False, "q1" -> True
        quant_bool = (quant == "q1")
    else:
        # Try to parse as boolean
        quant_bool = str(quant).lower() in ('true', '1', 'yes', 'on')
    
    quant_str_for_csv = "True" if quant_bool else "False"
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_size', 'seq_len', 'strategy', 'mesh_shape', 'dtype', 'quant'])
        for batch_size in batch_sizes:
            writer.writerow([batch_size, seq_len, strategy, mesh_shape, dtype, quant_str_for_csv])
    
    print(f"Created sdy_gen.csv at {output_path} with {len(batch_sizes)} entries")
    print(f"  Format: dtype={dtype}, quant={quant_str_for_csv} (from input: {quant})")


def run_sdy_generator_batch(vinveli_home: Path, sdy_gen_csv: Path, max_workers: int = 1, 
                            strategy: str = None, dtype: str = None, quant: str = None):
    """Run run_sdy_generator_batch.py to generate MLIR files."""
    print(f"\n{'='*60}")
    print("Step 1: Generating original MLIR files using run_sdy_generator_batch.py")
    print(f"{'='*60}")
    
    # Debug: Print CSV contents to verify format
    print(f"\nDebug: Verifying CSV file contents...")
    with open(sdy_gen_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"  CSV has {len(rows)} rows")
        if rows:
            print(f"  CSV columns: {reader.fieldnames}")
            print(f"  First row: {rows[0]}")
            if len(rows) > 1:
                print(f"  Last row: {rows[-1]}")
    
    cmd = [
        sys.executable,
        'run_sdy_generator_batch.py',
        '--max-workers', str(max_workers),
        '--mode', 'csv_file',
        f'--csv_path={sdy_gen_csv.absolute()}',
        'system_config.yaml',
        'model_configs/deepseek_r1.yaml'
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Working directory: {vinveli_home}")
    print(f"CSV file: {sdy_gen_csv.absolute()}")
    
    result = subprocess.run(
        cmd,
        cwd=vinveli_home,
        capture_output=True,
        text=True
    )
    
    # Always print output for debugging
    if result.stdout:
        print(f"\nSTDOUT from run_sdy_generator_batch.py:")
        # Print last 100 lines to avoid overwhelming output
        stdout_lines = result.stdout.split('\n')
        if len(stdout_lines) > 100:
            print("  ... (showing last 100 lines) ...")
            for line in stdout_lines[-100:]:
                print(f"  {line}")
        else:
            for line in stdout_lines:
                print(f"  {line}")
    
    if result.stderr:
        print(f"\nSTDERR from run_sdy_generator_batch.py:")
        stderr_lines = result.stderr.split('\n')
        if len(stderr_lines) > 100:
            print("  ... (showing last 100 lines) ...")
            for line in stderr_lines[-100:]:
                print(f"  {line}")
        else:
            for line in stderr_lines:
                print(f"  {line}")
    
    if result.returncode != 0:
        print(f"\n❌ Error: run_sdy_generator_batch.py failed with return code {result.returncode}")
        raise RuntimeError(f"run_sdy_generator_batch.py failed with return code {result.returncode}")
    
    # Wait longer for files to be written and copied from container
    print("\nWaiting for MLIR files to be written...")
    import time
    time.sleep(5)  # Wait 5 seconds for file operations to complete
    
    # Verify that MLIR files actually exist
    print("\nVerifying generated MLIR files...")
    hlo_path = vinveli_home / "hlo"
    # Use provided parameters or infer from CSV
    if quant is None or dtype is None or strategy is None:
        # Read from CSV to get the values
        with open(sdy_gen_csv, 'r') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row:
                strategy = strategy or first_row.get('strategy', 'decode')
                dtype = dtype or first_row.get('dtype', 'fp8')
                quant_val = quant or first_row.get('quant', 'False')
            else:
                strategy = strategy or 'decode'
                dtype = dtype or 'fp8'
                quant_val = quant or 'False'
    else:
        quant_val = quant
    
    quant_bool = str(quant_val).lower() in ('true', '1', 'yes', 'on') if isinstance(quant_val, str) else bool(quant_val)
    quant_str_csv = "q1" if quant_bool else "q0"
    prefix = "hlo_pre" if strategy == "prefill" else "hlo_dec"
    
    # Read the CSV to check which files should exist
    with open(sdy_gen_csv, 'r') as f:
        reader = csv.DictReader(f)
        expected_files = []
        for row in reader:
            batch_size = int(row['batch_size'])
            seq_len = int(row['seq_len'])
            mesh_shape = row['mesh_shape']
            dtype = row.get('dtype', 'fp8')
            quant_val = row.get('quant', 'False')
            quant_bool = str(quant_val).lower() in ('true', '1', 'yes', 'on')
            quant_str_csv = "q1" if quant_bool else "q0"
            
            # Use the quant_str from CSV row, not the one we calculated
            quant_row_val = row.get('quant', 'False')
            quant_row_bool = str(quant_row_val).lower() in ('true', '1', 'yes', 'on')
            quant_row_str = "q1" if quant_row_bool else "q0"
            hlo_dir_name = f"{prefix}_deepseek_r1_{quant_row_str}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}"
            hlo_dir = hlo_path / hlo_dir_name
            mlir_file = hlo_dir / "deepseek_r1.mlir"
            
            expected_files.append({
                'batch_size': batch_size,
                'hlo_dir': hlo_dir,
                'mlir_file': mlir_file,
                'exists': mlir_file.exists()
            })
    
    # Report which files exist and which don't
    existing_files = [f for f in expected_files if f['exists']]
    missing_files = [f for f in expected_files if not f['exists']]
    
    print(f"  Expected {len(expected_files)} MLIR files")
    print(f"  Found {len(existing_files)} MLIR files")
    if missing_files:
        print(f"  ⚠️  Missing {len(missing_files)} MLIR files:")
        for f in missing_files[:10]:  # Show first 10
            dir_exists = f['hlo_dir'].exists()
            dir_status = "exists (empty)" if dir_exists else "missing"
            print(f"    - batch_size={f['batch_size']}: {f['mlir_file'].name} (directory {dir_status})")
        if len(missing_files) > 10:
            print(f"    ... and {len(missing_files) - 10} more")
    
    if len(existing_files) == 0:
        print(f"\n❌ Error: No MLIR files were generated. Cannot proceed.")
        print(f"   Check the STDOUT/STDERR above for generation errors.")
        print(f"\n   Common issues:")
        print(f"   - Decode mode with sequence parallelism (SP > 1) may fail because decode uses seq_len=1")
        print(f"   - Batch sizes incompatible with mesh_shape (e.g., batch_size=1 with mesh_shape=2x72x1)")
        print(f"   - Mesh axis sizes must evenly divide tensor dimensions")
        raise RuntimeError("No MLIR files were generated")
    elif len(missing_files) > 0:
        print(f"\n⚠️  Warning: {len(missing_files)} MLIR files failed to generate, but proceeding with {len(existing_files)} available files")
        print(f"   Failed batch sizes: {[f['batch_size'] for f in missing_files]}")
        print(f"   This may be due to incompatibility between batch_size and mesh_shape")
    
    print("\n✓ Original MLIR files generated successfully")


def find_mlir_file(vinveli_home: Path, batch_size: int, seq_len: int, strategy: str,
                   mesh_shape: str, dtype: str, quant: str) -> Optional[Path]:
    """Find MLIR file for given parameters.
    
    Tries multiple patterns to find the file:
    1. New format with dtype/quant: hlo_{pre|dec}_deepseek_r1_{quant}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}
    2. Old format without dtype/quant: hlo_{pre|dec}_deepseek_r1_{batch_size}_{seq_len}_{mesh_shape}
    3. Patterns with wildcards or suffixes (e.g., _pid12345)
    
    Args:
        vinveli_home: VINVELI_HOME directory (MLIR files are searched in vinveli_home/hlo/)
    """
    hlo_path = vinveli_home / "hlo"
    print(f"  [DEBUG] Searching for MLIR file in: {hlo_path}")
    print(f"  [DEBUG]   batch_size={batch_size}, seq_len={seq_len}, strategy={strategy}")
    print(f"  [DEBUG]   mesh_shape={mesh_shape}, dtype={dtype}, quant={quant}")
    quant_str = quant if quant.startswith('q') else f"q{quant}"
    prefix = "hlo_pre" if strategy == "prefill" else "hlo_dec"
    
    # Pattern 1: New format with dtype and quant (exact match)
    hlo_dir_name_with_dtype = f"{prefix}_deepseek_r1_{quant_str}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}"
    hlo_dir = hlo_path / hlo_dir_name_with_dtype
    if hlo_dir.exists():
        mlir_file = hlo_dir / "deepseek_r1.mlir"
        if mlir_file.exists():
            return mlir_file
    
    # Pattern 2: New format with dtype/quant (glob with suffix, e.g., _pid12345)
    matching_dirs = list(hlo_path.glob(f"{hlo_dir_name_with_dtype}*"))
    for hlo_dir in matching_dirs:
        mlir_file = hlo_dir / "deepseek_r1.mlir"
        if mlir_file.exists():
            return mlir_file
    
    # Pattern 3: Old format without dtype/quant (exact match)
    hlo_dir_name_simple = f"{prefix}_deepseek_r1_{batch_size}_{seq_len}_{mesh_shape}"
    hlo_dir = hlo_path / hlo_dir_name_simple
    if hlo_dir.exists():
        mlir_file = hlo_dir / "deepseek_r1.mlir"
        if mlir_file.exists():
            return mlir_file
    
    # Pattern 4: Old format without dtype/quant (glob with suffix)
    matching_dirs = list(hlo_path.glob(f"{hlo_dir_name_simple}*"))
    for hlo_dir in matching_dirs:
        mlir_file = hlo_dir / "deepseek_r1.mlir"
        if mlir_file.exists():
            return mlir_file
    
    # Pattern 5: Very flexible pattern - match batch_size, seq_len, and mesh_shape anywhere
    # This handles cases where the format might be slightly different
    flexible_pattern = f"{prefix}_deepseek_r1_*_{batch_size}_*_{seq_len}_*{mesh_shape}*"
    matching_dirs = list(hlo_path.glob(flexible_pattern))
    for hlo_dir in matching_dirs:
        # Verify it actually contains our batch_size, seq_len, and mesh_shape
        dir_name = hlo_dir.name
        if (f"_{batch_size}_" in dir_name and 
            f"_{seq_len}_" in dir_name and 
            mesh_shape in dir_name):
            mlir_file = hlo_dir / "deepseek_r1.mlir"
            if mlir_file.exists():
                return mlir_file
    
    return None


def find_stats_csv_files(stats_dir: Path, batch_size: int, seq_len: int,
                        strategy: str, mesh_shape: str, dtype: str, quant: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find comm_stats.csv and comp_stats.csv files for a given batch_size.
    
    Returns:
        Tuple of (comm_stats_csv_path, comp_stats_csv_path) or (None, None) if not found
    """
    quant_str = quant if quant.startswith('q') else f"q{quant}"
    prefix = "hlo_pre" if strategy == "prefill" else "hlo_dec"
    hlo_dir_name = f"{prefix}_deepseek_r1_{quant_str}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}"
    
    # Search for CSV files in subdirectories matching the MLIR file pattern
    comm_stats = None
    comp_stats = None
    
    # Try exact match first
    exact_dir = stats_dir / hlo_dir_name
    if exact_dir.exists():
        comm_file = exact_dir / "comm_stats.csv"
        comp_file = exact_dir / "comp_stats.csv"
        if comm_file.exists():
            comm_stats = comm_file
        if comp_file.exists():
            comp_stats = comp_file
    
    # Recursively search if not found
    if comm_stats is None or comp_stats is None:
        for subdir in stats_dir.rglob(hlo_dir_name):
            if subdir.is_dir():
                comm_file = subdir / "comm_stats.csv"
                comp_file = subdir / "comp_stats.csv"
                if comm_file.exists() and comm_stats is None:
                    comm_stats = comm_file
                if comp_file.exists() and comp_stats is None:
                    comp_stats = comp_file
    
    # Also try searching by batch_size in directory names
    if comm_stats is None or comp_stats is None:
        for comm_file in stats_dir.rglob("comm_stats.csv"):
            parent = comm_file.parent
            if str(batch_size) in parent.name or hlo_dir_name in parent.name:
                comm_stats = comm_file
                break
        
        for comp_file in stats_dir.rglob("comp_stats.csv"):
            parent = comp_file.parent
            if str(batch_size) in parent.name or hlo_dir_name in parent.name:
                comp_stats = comp_file
                break
    
    return comm_stats, comp_stats


def create_vinveli_config_from_model_config(
    model_config_path: Path, 
    seq_len: int, 
    mesh_shape: str,
    output_config_path: Path
) -> Path:
    """Create a temporary vinveli config file with modify_batch_size section from model config.
    
    Args:
        model_config_path: Path to model config YAML file
        seq_len: Sequence length (may override model config)
        mesh_shape: Mesh shape string (e.g., "1x48x1") to extract SP and EP
        output_config_path: Path where vinveli config will be written
    
    Returns:
        Path to created vinveli config file
    """
    # Load model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    if model_config is None:
        raise ValueError(f"Model config file is empty: {model_config_path}")
    
    # Extract SP and EP from mesh_shape
    mesh_parts = mesh_shape.split('x')
    if len(mesh_parts) < 2:
        raise ValueError(f"Invalid mesh_shape format: {mesh_shape}. Expected format: 'NxMxK'")
    
    sp = int(mesh_parts[0])  # Sequence parallelism (first dimension)
    ep = int(mesh_parts[1])   # Expert parallelism (second dimension)
    
    # Create vinveli config structure
    vinveli_config = {
        'modify_batch_size': {
            'num_experts': int(model_config.get('num_experts', 0)),
            'seq_len': seq_len,  # Use provided seq_len (may override model config)
            'num_experts_per_tok': int(model_config.get('num_experts_per_tok', 0)),
            'sp': sp,
        }
    }
    
    # Add num_heads if present in model config
    if 'num_heads' in model_config and model_config['num_heads']:
        vinveli_config['modify_batch_size']['num_heads'] = int(model_config['num_heads'])
    
    # Write vinveli config to temporary file
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, 'w') as f:
        yaml.dump(vinveli_config, f, default_flow_style=False)
    
    print(f"  Created temporary vinveli config: {output_config_path}")
    print(f"    num_experts: {vinveli_config['modify_batch_size']['num_experts']}")
    print(f"    seq_len: {vinveli_config['modify_batch_size']['seq_len']}")
    print(f"    num_experts_per_tok: {vinveli_config['modify_batch_size']['num_experts_per_tok']}")
    print(f"    sp: {sp} (from mesh_shape)")
    if 'num_heads' in vinveli_config['modify_batch_size']:
        print(f"    num_heads: {vinveli_config['modify_batch_size']['num_heads']}")
    
    return output_config_path


def resolve_batch_size_modifier_binary(args) -> Path:
    """Locate the XLA //xla/service/gpu/model:batch_size_modifier binary."""
    if args.batch_size_modifier_bin:
        p = Path(args.batch_size_modifier_bin)
        if not p.is_file():
            raise FileNotFoundError(
                f"--batch-size-modifier-bin not found or not a file: {p}")
        return p.resolve()
    env_bin = os.environ.get("BATCH_SIZE_MODIFIER_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"BATCH_SIZE_MODIFIER_BIN is set but not a file: {env_bin}")
    xla_root = args.xla_root or os.environ.get("XLA_ROOT")
    if xla_root:
        cand = Path(xla_root) / "bazel-bin/xla/service/gpu/model/batch_size_modifier"
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        "batch_size_modifier binary not found. Build with:\n"
        "  bazel build //xla/service/gpu/model:batch_size_modifier\n"
        "Then set --batch-size-modifier-bin, BATCH_SIZE_MODIFIER_BIN, or "
        "--xla-root / XLA_ROOT so bazel-bin/.../batch_size_modifier can be found.")


def modify_mlir_file(modifier_bin: Path, input_mlir: Path, output_mlir: Path,
                     old_batch_size: int, new_batch_size: int,
                     model_config_path: Path, strategy: str, seq_len: int,
                     mesh_shape: str,
                     comm_stats_csv: Optional[Path] = None,
                     comp_stats_csv: Optional[Path] = None,
                     vinveli_home: Optional[Path] = None):
    """Run XLA batch_size_modifier to rewrite MLIR (strategy is unused; kept for API parity).

    Args:
        modifier_bin: Path to the batch_size_modifier executable.
        model_config_path: Path to model config YAML (converted to vinveli-style temp config).
        comm_stats_csv / comp_stats_csv: Required CSV paths (same as Python workflow).
        vinveli_home: Unused (retained for call-site compatibility).
    """
    _ = vinveli_home
    _ = strategy

    # CSV files are required
    if comm_stats_csv is None or not comm_stats_csv.exists():
        raise FileNotFoundError(f"comm_stats.csv is required but not found: {comm_stats_csv}")
    if comp_stats_csv is None or not comp_stats_csv.exists():
        raise FileNotFoundError(f"comp_stats.csv is required but not found: {comp_stats_csv}")
    
    # Ensure output directory exists
    output_mlir.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary vinveli config file with modify_batch_size section
    # Use a temporary file in the same directory as output_mlir
    temp_config_path = output_mlir.parent / f"temp_vinveli_config_{old_batch_size}_{new_batch_size}.yaml"
    try:
        vinveli_config_path = create_vinveli_config_from_model_config(
            model_config_path, seq_len, mesh_shape, temp_config_path
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create vinveli config file: {e}")
    
    cmd = [
        str(modifier_bin),
        '--input', str(input_mlir),
        '--output', str(output_mlir),
        '--old-batch-size', str(old_batch_size),
        '--new-batch-size', str(new_batch_size),
        '--config', str(vinveli_config_path),
        '--mesh-inference-path', str(input_mlir),
        '--comm-stats-csv', str(comm_stats_csv),
        '--comp-stats-csv', str(comp_stats_csv),
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    # Clean up temporary config file
    try:
        if temp_config_path.exists():
            temp_config_path.unlink()
    except Exception:
        pass  # Ignore cleanup errors
    
    if result.returncode != 0:
        print(f"  Error modifying MLIR file:")
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(
            f"batch_size_modifier ({modifier_bin.name}) failed with return code "
            f"{result.returncode}")
    
    if not output_mlir.exists():
        raise FileNotFoundError(f"Modified MLIR file was not created: {output_mlir}")


def copy_mlir_to_container(mlir_file: Path, xla_container_path: str, container_name: str = "xla") -> str:
    """Copy MLIR file to XLA container."""
    container_dir = f"{xla_container_path}/{mlir_file.parent.name}"
    container_file_path = f"{container_dir}/{mlir_file.name}"
    
    # Create directory in container
    result = subprocess.run(
        ["docker", "exec", container_name, "mkdir", "-p", container_dir],
        check=True,
        capture_output=True,
        text=True
    )
    
    # Copy file to container
    result = subprocess.run(
        ["docker", "cp", str(mlir_file), f"{container_name}:{container_file_path}"],
        check=True,
        capture_output=True,
        text=True
    )
    
    return container_file_path


def copy_mlir_files_from_csv(sdy_gen_csv: Path, mlir_files: Dict[int, Path], 
                             xla_container_path: str, container_name: str = "xla") -> List[str]:
    """Copy MLIR files to container based on CSV entries.
    
    Checks if files exist before copying and reports any missing files.
    """
    container_paths = []
    missing_files = []
    
    with open(sdy_gen_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_size = int(row['batch_size'])
            if batch_size in mlir_files:
                mlir_file = mlir_files[batch_size]
                # Check if file exists before copying
                if not mlir_file.exists():
                    missing_files.append((batch_size, mlir_file))
                    print(f"  ⚠️  Warning: MLIR file not found for batch_size={batch_size}: {mlir_file}")
                    continue
                
                container_path = copy_mlir_to_container(mlir_file, xla_container_path, container_name)
                container_paths.append(container_path)
                print(f"  ✓ Copied: {mlir_file.name} -> {container_path}")
            else:
                missing_files.append((batch_size, None))
                print(f"  ⚠️  Warning: No MLIR file mapping found for batch_size={batch_size}")
    
    if missing_files:
        print(f"\n  ⚠️  Total missing files: {len(missing_files)}")
    
    return container_paths


def run_analytical_latency_batch(vinveli_home: Path, sdy_gen_csv: Path, stats_dir_name: str,
                                hardware_arch: str, overlap_factor: float):
    """Run run_analytical_latency_batch.py to calculate latency.
    
    Note: 
    - stats_dir_name should be a directory name (not an absolute path),
      as run_analytical_latency_batch.py expects a relative directory name.
    - run_analytical_latency_batch.py reads the CSV file and finds MLIR files
      in VINVELI_HOME/hlo/ based on the CSV entries (batch_size, seq_len, strategy, etc.)
    - The MLIR files must be in VINVELI_HOME/hlo/ - there is no option to use an alternate path
    """
    print(f"\n[DEBUG] run_analytical_latency_batch will search for MLIR files in: {vinveli_home / 'hlo'}")
    print(f"[DEBUG]   It reads the CSV file and constructs MLIR paths based on CSV entries")
    print(f"[DEBUG]   CSV file: {sdy_gen_csv}")
    sdy_gen_abs_path = sdy_gen_csv.absolute()
    
    cmd = [
        sys.executable,
        'run_analytical_latency_batch.py',
        '--csv-file', str(sdy_gen_abs_path),
        '--model-config', 'model_configs/deepseek_r1.yaml',
        '--system-config', 'system_config.yaml',
        '--stats-dir', stats_dir_name,  # Directory name, not absolute path
        '--hardware-architectures', hardware_arch,
        '--overlap-factor', str(overlap_factor)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {vinveli_home}")
    
    result = subprocess.run(
        cmd,
        cwd=vinveli_home,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running run_analytical_latency_batch.py:")
        print(f"STDOUT: {result.stdout[-1000:]}")
        print(f"STDERR: {result.stderr[-1000:]}")
        raise RuntimeError(f"run_analytical_latency_batch.py failed with return code {result.returncode}")
    
    print("✓ Analytical latency batch processing completed")


def extract_latency_from_stats(stats_dir: Path, batch_size: int, seq_len: int, 
                               strategy: str, mesh_shape: str, dtype: str, quant: str) -> Dict[str, float]:
    """Extract latency from stats directory created by run_analytical_latency_batch.py, grouped by hw_arch.
    
    run_analytical_latency_batch.py creates stats in subdirectories matching MLIR file names.
    Structure: stats_dir/{hlo_dir_name}/device_stats.csv
    We search recursively to find the matching device_stats.csv file.
    
    Returns:
        Dictionary mapping hw_arch to latency in milliseconds, or empty dict if not found
    """
    quant_str = quant if quant.startswith('q') else f"q{quant}"
    prefix = "hlo_pre" if strategy == "prefill" else "hlo_dec"
    
    # The stats directory structure from run_analytical_latency_batch.py
    # Look for directory matching the MLIR file pattern
    hlo_dir_name = f"{prefix}_deepseek_r1_{quant_str}_{dtype}_{batch_size}_{seq_len}_{mesh_shape}"
    
    # First try exact match in subdirectory
    exact_match = stats_dir / hlo_dir_name / "device_stats.csv"
    if exact_match.exists():
        try:
            stats_df = pd.read_csv(exact_match)
            return _extract_latency_by_hw_arch_from_dataframe(stats_df)
        except Exception as e:
            print(f"    Warning: Failed to read {exact_match}: {e}")
    
    # Recursively search for device_stats.csv files
    # run_analytical_latency_batch.py may create nested directory structures
    for device_stats_file in stats_dir.rglob("device_stats.csv"):
        # Check if this file is in a directory that matches our batch_size pattern
        parent_dir = device_stats_file.parent.name
        if str(batch_size) in parent_dir or hlo_dir_name in parent_dir:
            try:
                stats_df = pd.read_csv(device_stats_file)
                latencies = _extract_latency_by_hw_arch_from_dataframe(stats_df)
                if latencies:
                    return latencies
            except Exception as e:
                continue
    
    # If no match found, try any device_stats.csv (for cases where structure is different)
    for device_stats_file in stats_dir.rglob("device_stats.csv"):
        try:
            stats_df = pd.read_csv(device_stats_file)
            latencies = _extract_latency_by_hw_arch_from_dataframe(stats_df)
            if latencies:
                return latencies
        except Exception as e:
            continue
    
    return {}


def _find_actual_stats_dir(vinveli_home: Path, stats_dir_name: str, fallback_path: Path) -> Path:
    """Find where run_analytical_latency_batch.py actually wrote the stats.
    
    The script may write to:
    1. Current working directory (vinveli_home) / stats_dir_name
    2. Default stats_path from config / stats_dir_name
    3. The fallback_path we provided
    
    Returns the first location that exists and contains device_stats.csv files.
    """
    # Try current working directory first
    cwd_stats = vinveli_home / stats_dir_name
    if cwd_stats.exists():
        # Check if it has any device_stats.csv files
        if list(cwd_stats.rglob("device_stats.csv")):
            return cwd_stats
    
    # Try fallback path
    if fallback_path.exists():
        if list(fallback_path.rglob("device_stats.csv")):
            return fallback_path
    
    # Try to find stats_dir_name in vinveli_home or subdirectories
    for potential_dir in vinveli_home.rglob(stats_dir_name):
        if potential_dir.is_dir() and list(potential_dir.rglob("device_stats.csv")):
            return potential_dir
    
    # Return fallback path even if empty (for error reporting)
    return fallback_path


def _extract_latency_by_hw_arch_from_dataframe(stats_df: pd.DataFrame) -> Dict[str, float]:
    """Extract latency values from a stats DataFrame, grouped by hw_arch.
    
    Returns:
        Dictionary mapping hw_arch to latency in milliseconds
    """
    if len(stats_df) == 0:
        return {}
    
    # Check if hw_arch column exists
    if 'hw_arch' not in stats_df.columns:
        # Fallback: try to extract single latency value (for backward compatibility)
        latency_col = None
        for col in ['overlapped_latency_secs', 'latency_secs', 'original_latency_secs', 'latency']:
            if col in stats_df.columns:
                latency_col = col
                break
        
        if latency_col:
            latency_secs = stats_df[latency_col].values[0]
            # Convert to milliseconds if in seconds
            if latency_col.endswith('_secs') or latency_secs < 1000:
                latency_ms = latency_secs * 1000
            else:
                latency_ms = latency_secs
            # Return with a default hw_arch key
            return {'unknown': latency_ms}
        return {}
    
    # Group by hw_arch and aggregate latencies
    latencies_by_arch = {}
    
    # Try different latency column names
    latency_col = None
    for col in ['overlapped_latency_secs', 'latency_secs', 'original_latency_secs', 'latency']:
        if col in stats_df.columns:
            latency_col = col
            break
    
    if not latency_col:
        return {}
    
    # Group by hw_arch and sum latencies (in case there are multiple devices per arch)
    for hw_arch in stats_df['hw_arch'].unique():
        arch_df = stats_df[stats_df['hw_arch'] == hw_arch]
        # Sum latencies for all devices with this hw_arch
        total_latency_secs = arch_df[latency_col].sum()
        # Convert to milliseconds if in seconds
        if latency_col.endswith('_secs') or total_latency_secs < 1000:
            latency_ms = total_latency_secs * 1000
        else:
            latency_ms = total_latency_secs
        latencies_by_arch[hw_arch] = latency_ms
    
    return latencies_by_arch


def _extract_latency_from_dataframe(stats_df: pd.DataFrame) -> Optional[float]:
    """Extract latency value from a stats DataFrame (backward compatibility)."""
    latencies = _extract_latency_by_hw_arch_from_dataframe(stats_df)
    if latencies:
        # Return the first value (or sum if multiple)
        return sum(latencies.values())
    return None


def main():
    """Main function."""
    args = parse_args()
    
    # Get paths
    vinveli_home = get_vinveli_home(args)
    config_path = get_model_config_path(args)
    try:
        modifier_bin = resolve_batch_size_modifier_binary(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    print(f"\nUsing batch_size_modifier binary: {modifier_bin}")
    
    # Generate batch sizes
    batch_sizes = generate_batch_sizes(args.max_batch_size, num_files=args.num_datapoints)
    print(f"\nGenerated batch sizes ({args.num_datapoints} datapoints): {batch_sizes}")
    
    # Warn about potential incompatibilities
    mesh_parts = args.mesh_shape.split('x')
    if len(mesh_parts) >= 1:
        sp = int(mesh_parts[0])  # Sequence parallelism
        if sp > 1 and args.strategy == 'decode':
            print(f"\n⚠️  Warning: Decode mode with sequence parallelism (SP={sp}) may fail.")
            print(f"   Decode uses seq_len=1, which may not be divisible by SP={sp}.")
            print(f"   Some batch sizes may fail to generate MLIR files.")
            print(f"   The script will continue with successfully generated files.")
    
    # Create temporary directory for outputs
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="batch_size_validation_"))
        print(f"Using temporary output directory: {output_dir}")
    
    # Step 0: Clear existing MLIR files (unless skipped)
    if not args.skip_cleanup:
        clear_mlir_files(vinveli_home, batch_sizes, args.seq_len, args.strategy,
                        args.mesh_shape, args.dtype, args.quant)
    else:
        print(f"\n{'='*60}")
        print("Step 0: Skipping MLIR file cleanup (--skip-cleanup flag set)")
        print(f"{'='*60}")
    
    # Step 1: Generate original MLIR files
    sdy_gen_csv_original = output_dir / "sdy_gen_original.csv"
    create_sdy_gen_csv(batch_sizes, args.seq_len, args.strategy, args.mesh_shape,
                      args.dtype, args.quant, sdy_gen_csv_original)
    
    try:
        run_sdy_generator_batch(vinveli_home, sdy_gen_csv_original, args.max_workers,
                               strategy=args.strategy, dtype=args.dtype, quant=args.quant)
    except Exception as e:
        print(f"Error generating original MLIR files: {e}")
        return 1
    
    # Wait a moment for files to be written (in case of async operations)
    import time
    time.sleep(2)
    
    # Debug: Check what files were actually generated for our specific parameters
    hlo_path = vinveli_home / "hlo"
    quant_str = args.quant if args.quant.startswith('q') else f"q{args.quant}"
    prefix = "hlo_pre" if args.strategy == "prefill" else "hlo_dec"
    
    # Check for files matching our expected pattern
    expected_pattern_base = f"{prefix}_deepseek_r1_{quant_str}_{args.dtype}_*_{args.seq_len}_{args.mesh_shape}*"
    matching_dirs = list(hlo_path.glob(expected_pattern_base))
    
    print(f"\n  Debug: Searching for generated files...")
    print(f"    Expected pattern: {prefix}_deepseek_r1_{quant_str}_{args.dtype}_<batch_size>_{args.seq_len}_{args.mesh_shape}*")
    print(f"    Found {len(matching_dirs)} directories matching pattern")
    if matching_dirs:
        print(f"    Sample matches (first 5):")
        for d in matching_dirs[:5]:
            print(f"      - {d.name}")
    else:
        # Try old format (without dtype/quant)
        old_pattern = f"{prefix}_deepseek_r1_*_{args.seq_len}_{args.mesh_shape}*"
        old_matching = list(hlo_path.glob(old_pattern))
        print(f"    ⚠️  No files found with new format (dtype/quant)")
        if old_matching:
            print(f"    Found {len(old_matching)} directories with old format (without dtype/quant):")
            for d in old_matching[:5]:
                print(f"      - {d.name}")
            print(f"    This suggests run_sdy_generator_batch.py may not be using dtype/quant columns")
        else:
            print(f"    ⚠️  No files found matching any pattern. Check if generation succeeded.")
    
    # Find all original MLIR files
    # Use the verification results from above if available, otherwise search
    original_mlir_files = {}
    hlo_path = vinveli_home / "hlo"
    print(f"\n{'='*60}")
    print(f"Step 2: Finding original MLIR files")
    print(f"{'='*60}")
    print(f"VINVELI_HOME: {vinveli_home}")
    print(f"Searching for MLIR files in: {hlo_path}")
    print(f"Note: run_sdy_generator_batch.py always saves MLIR files to VINVELI_HOME/hlo/")
    print(f"      There is no option to change this output path.\n")
    
    for batch_size in batch_sizes:
        mlir_file = find_mlir_file(vinveli_home, batch_size, args.seq_len,
                                   args.strategy, args.mesh_shape, args.dtype, args.quant)
        if mlir_file and mlir_file.exists():
            # Double-check the file actually exists and has content
            if mlir_file.stat().st_size > 0:
                original_mlir_files[batch_size] = mlir_file
                print(f"  ✓ Found MLIR file for batch_size={batch_size}: {mlir_file.name} ({mlir_file.stat().st_size} bytes)")
            else:
                print(f"  ⚠️  Warning: MLIR file exists but is empty for batch_size={batch_size}: {mlir_file}")
        else:
            # Enhanced error message with expected pattern
            quant_str = args.quant if args.quant.startswith('q') else f"q{args.quant}"
            prefix = "hlo_pre" if args.strategy == "prefill" else "hlo_dec"
            expected_pattern = f"{prefix}_deepseek_r1_{quant_str}_{args.dtype}_{batch_size}_{args.seq_len}_{args.mesh_shape}"
            print(f"  ⚠️  Warning: Original MLIR file not found for batch_size={batch_size}")
            print(f"     Expected pattern: {expected_pattern}*")
            print(f"     Searched in: {hlo_path}")
            
            # Check if directory exists but is empty
            hlo_dir = hlo_path / expected_pattern
            if hlo_dir.exists():
                print(f"     Directory exists but is empty or MLIR file missing")
                dir_contents = list(hlo_dir.iterdir())
                if dir_contents:
                    print(f"     Directory contains: {[f.name for f in dir_contents]}")
    
    # Step 2: Run analytical latency on original files to get CSV stats (needed for modify batch size script)
    print(f"\n{'='*60}")
    print("Step 2: Running analytical latency for original MLIR files")
    print(f"{'='*60}")
    
    if not original_mlir_files:
        print("⚠️  Error: No original MLIR files were found. Cannot proceed with latency calculation.")
        print(f"\n   Attempted to generate {len(batch_sizes)} batch sizes: {batch_sizes}")
        print(f"   Successfully found: {len(original_mlir_files)} files")
        print(f"\n   Possible causes:")
        print(f"   - Batch sizes incompatible with mesh_shape={args.mesh_shape}")
        print(f"   - Decode mode with sequence parallelism (SP > 1) may fail")
        print(f"   - Check STDOUT/STDERR above for generation errors")
        return 1
    
    print(f"\n✓ Found {len(original_mlir_files)}/{len(batch_sizes)} MLIR files")
    if len(original_mlir_files) < len(batch_sizes):
        missing = set(batch_sizes) - set(original_mlir_files.keys())
        print(f"   Missing batch sizes: {sorted(missing)}")
        print(f"   Continuing with available batch sizes: {sorted(original_mlir_files.keys())}")
    
    # Create filtered CSV with only batch sizes that have MLIR files
    # This ensures run_analytical_latency_batch.py only processes files that exist
    sdy_gen_csv_original_filtered = output_dir / "sdy_gen_original_filtered.csv"
    with open(sdy_gen_csv_original, 'r') as f_in, open(sdy_gen_csv_original_filtered, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            batch_size = int(row['batch_size'])
            if batch_size in original_mlir_files:
                writer.writerow(row)
    
    if len(original_mlir_files) < len(batch_sizes):
        print(f"\nCreated filtered CSV with {len(original_mlir_files)} entries: {sdy_gen_csv_original_filtered.name}")
    else:
        # If all files were found, use the original CSV
        sdy_gen_csv_original_filtered = sdy_gen_csv_original
    
    print("\nCopying original MLIR files to container...")
    print(f"  Copying {len(original_mlir_files)} original MLIR files...")
    copy_mlir_files_from_csv(sdy_gen_csv_original_filtered, original_mlir_files,
                             args.xla_container_path, args.container_name)
    
    # Use directory name (not absolute path) for stats-dir
    stats_dir_original_name = "stats_original"
    stats_dir_original = output_dir / stats_dir_original_name
    stats_dir_original.mkdir(parents=True, exist_ok=True)
    
    try:
        run_analytical_latency_batch(vinveli_home, sdy_gen_csv_original_filtered, stats_dir_original_name,
                                    args.hardware_arch, args.overlap_factor)
    except Exception as e:
        print(f"Error running analytical latency for original files: {e}")
        return 1
    
    # Find where results were actually written
    stats_dir_original_actual = _find_actual_stats_dir(vinveli_home, stats_dir_original_name, stats_dir_original)
    print(f"\n✓ Original stats written to: {stats_dir_original_actual}")
    
    # Step 3: Generate modified MLIR files using CSV stats from Step 2
    print(f"\n{'='*60}")
    print("Step 3: Generating modified MLIR files using modify batch size script")
    print(f"{'='*60}")
    
    # Find reference MLIR file (with max_batch_size)
    # NOTE: max_batch_size is used as the reference (old_batch_size) for modification
    reference_mlir = find_mlir_file(vinveli_home, args.max_batch_size, args.seq_len,
                                    args.strategy, args.mesh_shape, args.dtype, args.quant)
    
    if not reference_mlir:
        print(f"⚠️  Error: Reference MLIR file not found for batch_size={args.max_batch_size}")
        print(f"   This file should have been generated in Step 1.")
        return 1
    
    print(f"✓ Found reference MLIR file: {reference_mlir}")
    print(f"  Using this as input to modify batch size script")
    print(f"  Reference batch_size (old_batch_size) = {args.max_batch_size}")
    
    # Find CSV stats files for the reference batch_size (max_batch_size)
    # These are required by the modify batch size script
    print(f"\nFinding CSV stats files for reference batch_size={args.max_batch_size}...")
    comm_stats_csv, comp_stats_csv = find_stats_csv_files(
        stats_dir_original_actual, args.max_batch_size, args.seq_len,
        args.strategy, args.mesh_shape, args.dtype, args.quant
    )
    
    if comm_stats_csv is None:
        print(f"⚠️  Error: comm_stats.csv not found for batch_size={args.max_batch_size}")
        print(f"   Searched in: {stats_dir_original_actual}")
        print(f"   This file is required by the modify batch size script")
        return 1
    
    if comp_stats_csv is None:
        print(f"⚠️  Error: comp_stats.csv not found for batch_size={args.max_batch_size}")
        print(f"   Searched in: {stats_dir_original_actual}")
        print(f"   This file is required by the modify batch size script")
        return 1
    
    print(f"✓ Found comm_stats.csv: {comm_stats_csv}")
    print(f"✓ Found comp_stats.csv: {comp_stats_csv}")
    print(f"  These will be passed to batch_size_modifier")
    
    modified_mlir_dir = output_dir / "modified_mlir"
    modified_mlir_dir.mkdir(parents=True, exist_ok=True)
    
    modified_mlir_files = {}
    for batch_size in batch_sizes:
        print(f"\nProcessing batch_size={batch_size}...")
        
        modified_mlir = modified_mlir_dir / f"modified_batch_{batch_size}.mlir"
        
        try:
            # Use max_batch_size as old_batch_size (reference) and current batch_size as new_batch_size
            # Pass the CSV files from the reference batch_size
            modify_mlir_file(modifier_bin, reference_mlir, modified_mlir,
                            args.max_batch_size, batch_size, config_path,
                            args.strategy, args.seq_len, args.mesh_shape,
                            comm_stats_csv, comp_stats_csv, vinveli_home)
            # Verify file was created
            if modified_mlir.exists():
                modified_mlir_files[batch_size] = modified_mlir
                print(f"  ✓ Generated modified MLIR: {modified_mlir}")
            else:
                print(f"  ⚠️  Warning: Modified MLIR file was not created: {modified_mlir}")
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to modify MLIR for batch_size={batch_size}: {e}")
            continue
    
    # Step 4: Create CSV for modified files and copy to container
    print(f"\n{'='*60}")
    print("Step 4: Running analytical latency for modified MLIR files")
    print(f"{'='*60}")
    
    if not modified_mlir_files:
        print("⚠️  Error: No modified MLIR files were generated. Cannot proceed with latency calculation.")
        return 1
    
    sdy_gen_csv_modified = output_dir / "sdy_gen_modified.csv"
    create_sdy_gen_csv(list(modified_mlir_files.keys()), args.seq_len, args.strategy,
                      args.mesh_shape, args.dtype, args.quant, sdy_gen_csv_modified)
    
    print("\nCopying modified MLIR files to container...")
    print(f"  Checking {len(modified_mlir_files)} modified MLIR files...")
    copy_mlir_files_from_csv(sdy_gen_csv_modified, modified_mlir_files,
                             args.xla_container_path, args.container_name)
    
    # Use directory name (not absolute path) for stats-dir
    stats_dir_modified_name = "stats_modified"
    stats_dir_modified = output_dir / stats_dir_modified_name
    stats_dir_modified.mkdir(parents=True, exist_ok=True)
    
    try:
        run_analytical_latency_batch(vinveli_home, sdy_gen_csv_modified, stats_dir_modified_name,
                                    args.hardware_arch, args.overlap_factor)
    except Exception as e:
        print(f"Error running analytical latency for modified files: {e}")
        return 1
    
    # Step 5: Extract and compare latencies
    print(f"\n{'='*60}")
    print("Step 5: Latency Comparison Results")
    print(f"{'='*60}")
    
    # results structure: {batch_size: {hw_arch: {'original': float, 'modified': float}}}
    results = {}
    
    # Find where modified stats were written
    stats_dir_modified_actual = _find_actual_stats_dir(vinveli_home, stats_dir_modified_name, stats_dir_modified)
    print(f"\nFound modified stats directory: {stats_dir_modified_actual}")
    
    # Extract latencies from original files (grouped by hw_arch)
    print(f"\nExtracting latencies from original files...")
    print(f"  Looking in: {stats_dir_original_actual}")
    for batch_size in batch_sizes:
        if batch_size in original_mlir_files:
            latencies_by_arch = extract_latency_from_stats(stats_dir_original_actual, batch_size, args.seq_len,
                                                           args.strategy, args.mesh_shape, args.dtype, args.quant)
            if latencies_by_arch:
                results[batch_size] = {}
                for hw_arch, latency in latencies_by_arch.items():
                    if hw_arch not in results[batch_size]:
                        results[batch_size][hw_arch] = {'original': None, 'modified': None}
                    results[batch_size][hw_arch]['original'] = latency
                    print(f"  batch_size={batch_size}, hw_arch={hw_arch}: Original latency = {latency:.4f} ms")
            else:
                print(f"  batch_size={batch_size}: Could not extract latency from stats")
    
    # Extract latencies from modified files (grouped by hw_arch)
    print(f"\nExtracting latencies from modified files...")
    print(f"  Looking in: {stats_dir_modified_actual}")
    for batch_size in modified_mlir_files.keys():
        if batch_size in results:
            latencies_by_arch = extract_latency_from_stats(stats_dir_modified_actual, batch_size, args.seq_len,
                                                           args.strategy, args.mesh_shape, args.dtype, args.quant)
            if latencies_by_arch:
                for hw_arch, latency in latencies_by_arch.items():
                    # Only compare with corresponding hw_arch from original
                    if hw_arch in results[batch_size]:
                        results[batch_size][hw_arch]['modified'] = latency
                        print(f"  batch_size={batch_size}, hw_arch={hw_arch}: Modified latency = {latency:.4f} ms")
                    else:
                        # If hw_arch not in original, add it
                        if hw_arch not in results[batch_size]:
                            results[batch_size][hw_arch] = {'original': None, 'modified': None}
                        results[batch_size][hw_arch]['modified'] = latency
                        print(f"  batch_size={batch_size}, hw_arch={hw_arch}: Modified latency = {latency:.4f} ms (no original to compare)")
            else:
                print(f"  batch_size={batch_size}: Could not extract latency from stats")
    
    # Print comparison table with hw_arch column
    print(f"\n{'Batch Size':<12} {'HW_Arch':<15} {'Original (ms)':>15} {'Modified (ms)':>15} {'Gap (ms)':>15} {'Gap (%)':>15}")
    print("-" * 100)
    
    all_gaps = []
    for batch_size in sorted(results.keys()):
        for hw_arch in sorted(results[batch_size].keys()):
            orig = results[batch_size][hw_arch].get('original')
            mod = results[batch_size][hw_arch].get('modified')
            
            # Only print rows where we have both original and modified (corresponding hw_arch)
            if orig is not None and mod is not None:
                gap = mod - orig
                gap_pct = (gap / orig * 100) if orig != 0 else 0
                all_gaps.append(gap)
                print(f"{batch_size:<12} {hw_arch:<15} {orig:>15.4f} {mod:>15.4f} {gap:>15.4f} {gap_pct:>15.2f}%")
            elif orig is not None:
                print(f"{batch_size:<12} {hw_arch:<15} {orig:>15.4f} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
            elif mod is not None:
                print(f"{batch_size:<12} {hw_arch:<15} {'N/A':>15} {mod:>15.4f} {'N/A':>15} {'N/A':>15}")
    
    # Summary statistics
    if all_gaps:
        print(f"\nSummary Statistics:")
        print(f"  Average gap: {sum(all_gaps)/len(all_gaps):.4f} ms")
        print(f"  Max gap: {max(all_gaps):.4f} ms")
        print(f"  Min gap: {min(all_gaps):.4f} ms")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Modified MLIR files: {modified_mlir_dir}")
    print(f"Original stats: {stats_dir_original}")
    print(f"Modified stats: {stats_dir_modified}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
