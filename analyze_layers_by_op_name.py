#!/usr/bin/env python3
"""
Count operations by op_name from comp_stats.csv or comm_stats.csv.

This script groups operations by their exact op_name and prints counts.
It can also group operations by layer (dense/expert) and display operations
for randomly selected layers.

Usage:
    python3 analyze_layers_by_op_name.py <csv_file> [--dense-layers N] [--expert-layers M] [--output FILE] [--flops-results FILE]
    
Arguments:
    csv_file        Path to comp_stats.csv or comm_stats.csv file
    --dense-layers  Number of dense layers (optional)
    --expert-layers Number of expert layers (optional)
    --output        Output JSON file path to write all layer operations (optional)
    --flops-results Path to flops_results.csv file with expected FLOPs (optional)
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

def parse_csv(csv_path: str) -> list:
    """Parse CSV file."""
    operations = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                operations.append(row)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    return operations

def count_by_op_name(operations: list) -> Counter:
    """Count operations by op_name."""
    op_name_counts = Counter()
    
    for op in operations:
        op_name = op.get('op_name', '')
        if op_name:
            op_name_counts[op_name] += 1
        else:
            op_name_counts['(empty)'] += 1
    
    return op_name_counts

def classify_op_names_by_count(counts: Counter, num_dense: int, num_expert: int) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Classify op_names based on their counts.
    
    Returns:
        (dense_only_op_names, expert_only_op_names, both_op_names)
    """
    dense_only = set()
    expert_only = set()
    both = set()
    
    total_layers = num_dense + num_expert
    
    for op_name, count in counts.items():
        if count == num_dense:
            dense_only.add(op_name)
        elif count == num_expert:
            expert_only.add(op_name)
        elif count == total_layers:
            both.add(op_name)
        # Otherwise, it's misc (count doesn't match expected pattern)
    
    return dense_only, expert_only, both

def assign_operations_to_layers(operations: list, num_dense: int, num_expert: int) -> Tuple[List[List[Dict]], List[List[Dict]], List[Dict]]:
    """
    Assign operations to layers based on op_name counts.
    
    Fills layers sequentially: all op_names for layer 0, then all op_names for layer 1, etc.
    
    Returns:
        (dense_layers, expert_layers, misc_ops)
        - dense_layers: 2D list [layer_index][operations]
        - expert_layers: 2D list [layer_index][operations]
        - misc_ops: list of operations that don't belong to any layer
    """
    # Initialize 2D arrays
    dense_layers = [[] for _ in range(num_dense)]
    expert_layers = [[] for _ in range(num_expert)]
    misc_ops = []
    
    # Count op_names first
    counts = count_by_op_name(operations)
    
    # Classify op_names
    dense_only_op_names, expert_only_op_names, both_op_names = classify_op_names_by_count(
        counts, num_dense, num_expert
    )
    
    print("=" * 100)
    print("OP_NAME CLASSIFICATION")
    print("=" * 100)
    print()
    print(f"Dense-only op_names ({len(dense_only_op_names)}):")
    for op_name in sorted(dense_only_op_names):
        print(f"  - {op_name} (count: {counts[op_name]})")
    print()
    
    print(f"Expert-only op_names ({len(expert_only_op_names)}):")
    for op_name in sorted(expert_only_op_names):
        print(f"  - {op_name} (count: {counts[op_name]})")
    print()
    
    print(f"Both dense and expert op_names ({len(both_op_names)}):")
    for op_name in sorted(both_op_names):
        print(f"  - {op_name} (count: {counts[op_name]})")
    print()
    
    # Track how many times we've seen each op_name (for sequential layer assignment)
    op_name_counters = defaultdict(int)
    
    # Process operations in order (CSV entries are already ordered)
    for op in operations:
        op_name = op.get('op_name', '')
        if not op_name:
            op_name = '(empty)'
        
        # Increment counter for this op_name
        op_name_counters[op_name] += 1
        occurrence_num = op_name_counters[op_name]
        
        if op_name in dense_only_op_names:
            # Assign nth occurrence to layer (n-1) % num_dense
            # This fills layer 0 with all first occurrences, layer 1 with all second occurrences, etc.
            layer_idx = (occurrence_num - 1) % num_dense
            dense_layers[layer_idx].append(op)
        
        elif op_name in expert_only_op_names:
            # Assign nth occurrence to layer (n-1) % num_expert
            layer_idx = (occurrence_num - 1) % num_expert
            expert_layers[layer_idx].append(op)
        
        elif op_name in both_op_names:
            if occurrence_num <= num_dense:
                # Assign first num_dense occurrences to dense layers
                layer_idx = (occurrence_num - 1) % num_dense
                dense_layers[layer_idx].append(op)
            else:
                # Assign remaining occurrences to expert layers
                # occurrence_num is now num_dense+1, num_dense+2, etc.
                expert_occurrence = occurrence_num - num_dense
                layer_idx = (expert_occurrence - 1) % num_expert
                expert_layers[layer_idx].append(op)
        
        else:
            # Misc operation
            misc_ops.append(op)
    
    return dense_layers, expert_layers, misc_ops

def print_op_name_counts(counts: Counter):
    """Print operation counts by op_name."""
    print("=" * 100)
    print("OPERATION COUNTS BY OP_NAME")
    print("=" * 100)
    print()
    
    if not counts:
        print("No operations found with op_name.")
        return
    
    # Sort by count (descending), then by op_name
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    
    print(f"{'Count':<10} {'op_name'}")
    print("-" * 100)
    
    for op_name, count in sorted_counts:
        print(f"{count:<10} {op_name}")
    
    print()
    print(f"Total unique op_names: {len(counts)}")
    print(f"Total operations: {sum(counts.values())}")
    print()

def get_flops_from_op(op: Dict) -> float:
    """Extract FLOPs from operation (convert from TFLOPs if available)."""
    try:
        tflops = op.get('tflops', '')
        if tflops:
            return float(tflops) * 1e12  # Convert TFLOPs to FLOPs
    except (ValueError, TypeError):
        pass
    return 0.0

def parse_flops_results_csv(csv_path: str) -> List[Dict]:
    """Parse flops_results.csv file."""
    flops_results = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip TOTAL row
                if row.get('Scope', '').upper() == 'TOTAL':
                    continue
                flops_results.append(row)
    except FileNotFoundError:
        print(f"Warning: Flops results file not found: {csv_path}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Warning: Error reading flops results file: {e}", file=sys.stderr)
        return []
    return flops_results

def extract_scope_and_einsum_from_op_name(op_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract scope and einsum from op_name.
    
    Examples:
        "jit(forward)/q_embed/btd,dr->btr/dot_general" -> ("q_embed", "btd,dr->btr")
        "jit(forward)/ffn/gate/btd,df->btf/dot_general" -> ("ffn/gate", "btd,df->btf")
        "jit(forward)/jax.lax.ragged_dot/routed_experts/we_gate/Te,ged->Td/dot_general" -> 
            ("jax.lax.ragged_dot/routed_experts/we_gate", "Te,ged->Td")
        "jit(forward)/...T,k->...Tk/dot_general" -> ("...T,k->...Tk", None)  # Special case
        "jit(forward)/ffn/moe_routed_expert/we_gate/jax.lax.ragged_dot/dot_general" ->
            ("jit(forward)/ffn/moe_routed_expert/we_gate", None)  # einsum is in Scope in CSV
    
    Returns:
        (scope, einsum) or (None, None) if not found
    """
    if not op_name:
        return None, None
    
    # Special case: op_name like "jit(forward)/...T,k->...Tk/dot_general"
    # The scope itself contains the einsum pattern
    if '/...T,k->...Tk/' in op_name or '/...T,k->...Tk/dot_general' in op_name:
        # Extract the part before /dot_general
        match = re.search(r'jit\(forward\)/(.+?)/dot_general', op_name)
        if match:
            return match.group(1), None  # Return scope, no einsum (will match by op_name)
    
    # Pattern for expert operations with full path: jit(forward)/jax.lax.ragged_dot/...
    expert_pattern = r'jit\(forward\)/(jax\.lax\.ragged_dot/[^/]+/[^/]+)/([^/]+)/dot_general'
    match = re.search(expert_pattern, op_name)
    if match:
        scope = match.group(1)  # Full path like "jax.lax.ragged_dot/routed_experts/we_gate"
        einsum = match.group(2)
        return scope, einsum
    
    # Pattern: jit(forward)/<scope>/<einsum>/dot_general
    # Handle cases like: jit(forward)/q_embed/btd,dr->btr/dot_general
    # Or: jit(forward)/ffn/gate/btd,df->btf/dot_general
    # Or: jit(forward)/ffn/moe_routed_expert/we_gate/Te,ged->Td/dot_general
    pattern = r'jit\(forward\)/([^/]+(?:/[^/]+)*)/([^/]+)/dot_general'
    match = re.search(pattern, op_name)
    
    if match:
        scope = match.group(1)
        einsum = match.group(2)
        
        # Special case: if einsum is "jax.lax.ragged_dot", it means the einsum is not in the op_name
        # but should be matched by the full scope path
        if einsum == "jax.lax.ragged_dot":
            # Return the full path as scope for matching
            full_scope = f"jit(forward)/{scope}"
            return full_scope, None
        
        return scope, einsum
    
    return None, None

def build_flops_lookup(flops_results: List[Dict]) -> Tuple[Dict[Tuple[str, str], Dict], Dict[str, Dict]]:
    """
    Build lookup dictionaries for matching operations to flops_results.
    
    Returns:
        (lookup_by_scope_einsum, lookup_by_op_name)
        - lookup_by_scope_einsum: (scope, einsum) -> flops_result row
        - lookup_by_op_name: op_name -> flops_result row (for cases where Scope is a full op_name)
    """
    lookup_by_scope_einsum = {}
    lookup_by_op_name = {}
    
    for row in flops_results:
        scope = row.get('Scope', '').strip()
        einsum = row.get('Einsum', '').strip()
        
        if scope and einsum:
            # Normal case: scope is a simple name like "q_embed", "attention", etc.
            lookup_by_scope_einsum[(scope, einsum)] = row
            
            # Special case: if scope looks like a full op_name (contains "jit(forward)"),
            # also index by the full op_name
            if 'jit(forward)' in scope:
                lookup_by_op_name[scope] = row
    
    return lookup_by_scope_einsum, lookup_by_op_name

def enrich_operations_with_flops(
    operations: List[Dict],
    lookup_by_scope_einsum: Dict[Tuple[str, str], Dict],
    lookup_by_op_name: Dict[str, Dict]
) -> Tuple[int, int, List[str]]:
    """
    Enrich operations with sharded_flops and flops_per_token from flops_results.
    Ensures exact one-to-one matching.
    
    Args:
        operations: List of operation dictionaries to enrich
        lookup_by_scope_einsum: Lookup dictionary from (scope, einsum) -> flops_result row
        lookup_by_op_name: Lookup dictionary from op_name -> flops_result row
    
    Returns:
        (matched_count, total_count, unmatched_ops)
    """
    matched_count = 0
    total_count = len(operations)
    unmatched_ops = []
    
    for op in operations:
        op_name = op.get('op_name', '')
        matched = False
        
        # Strategy 1: Try matching by full op_name (for cases where Scope in CSV is a full op_name)
        if op_name in lookup_by_op_name:
            row = lookup_by_op_name[op_name]
            op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
            op['expected_unsharded_flops'] = row.get('FLOPS', '')
            flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
            op['expected_flops_per_token'] = flops_per_token
            matched_count += 1
            matched = True
        else:
            # Strategy 2: Extract scope and einsum, try exact match
            scope, einsum = extract_scope_and_einsum_from_op_name(op_name)
            
            if scope:
                # If einsum is None, try matching by scope only (for cases like jax.lax.ragged_dot)
                if einsum is None:
                    # Try matching by full scope path in op_name lookup
                    if scope in lookup_by_op_name:
                        row = lookup_by_op_name[scope]
                        op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
                        op['expected_unsharded_flops'] = row.get('FLOPS', '')
                        flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
                        op['expected_flops_per_token'] = flops_per_token
                        matched_count += 1
                        matched = True
                    else:
                        # Try to find a unique match by scope in lookup_by_scope_einsum
                        # Only match if there's exactly one entry with this scope
                        matching_rows = [(s, e, r) for (s, e), r in lookup_by_scope_einsum.items() if s == scope]
                        if len(matching_rows) == 1:
                            _, _, row = matching_rows[0]
                            op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
                            op['expected_unsharded_flops'] = row.get('FLOPS', '')
                            flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
                            op['expected_flops_per_token'] = flops_per_token
                            matched_count += 1
                            matched = True
                elif einsum:
                    # Try exact match: (scope, einsum)
                    key = (scope, einsum)
                    if key in lookup_by_scope_einsum:
                        row = lookup_by_scope_einsum[key]
                        op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
                        op['expected_unsharded_flops'] = row.get('FLOPS', '')
                        flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
                        op['expected_flops_per_token'] = flops_per_token
                        matched_count += 1
                        matched = True
                    else:
                        # Strategy 3: Try matching with full "jit(forward)/" prefix
                        # Some CSV entries have Scope="jit(forward)/..." but extraction returns scope without prefix
                        full_scope = f"jit(forward)/{scope}"
                        key_with_full_scope = (full_scope, einsum)
                        if key_with_full_scope in lookup_by_scope_einsum:
                            row = lookup_by_scope_einsum[key_with_full_scope]
                            op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
                            op['expected_unsharded_flops'] = row.get('FLOPS', '')
                            flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
                            op['expected_flops_per_token'] = flops_per_token
                            matched_count += 1
                            matched = True
                        else:
                            # Strategy 4: Try matching by last part of scope if scope has multiple parts
                            # e.g., "ffn/gate" -> try "gate"
                            scope_parts = scope.split('/')
                            if len(scope_parts) > 1:
                                last_part = scope_parts[-1]
                                key_with_last_part = (last_part, einsum)
                                if key_with_last_part in lookup_by_scope_einsum:
                                    row = lookup_by_scope_einsum[key_with_last_part]
                                    op['expected_sharded_flops'] = row.get('Sharded_FLOPS', '')
                                    op['expected_unsharded_flops'] = row.get('FLOPS', '')
                                    flops_per_token = row.get('FLOPS_Per_Token`', '') or row.get('FLOPS_Per_Token', '')
                                    op['expected_flops_per_token'] = flops_per_token
                                    matched_count += 1
                                    matched = True
        
        if not matched:
            unmatched_ops.append(op_name)
    
    return matched_count, total_count, unmatched_ops

def write_layer_operations_to_file(
    dense_layers: List[List[Dict]],
    expert_layers: List[List[Dict]],
    misc_ops: List[Dict],
    output_path: str
):
    """
    Write all layer operations to a JSON file for consumption by other scripts.
    
    Args:
        dense_layers: 2D list [layer_index][operations]
        expert_layers: 2D list [layer_index][operations]
        misc_ops: list of misc operations
        output_path: path to output JSON file
    """
    output_data = {
        "dense_layers": [],
        "expert_layers": [],
        "misc_ops": {
            "operations": misc_ops,
            "total_flops": sum(get_flops_from_op(op) for op in misc_ops),
            "total_expected_flops": sum(get_expected_flops_from_op(op) for op in misc_ops),
            "total_flops_per_token": sum(get_expected_flops_per_token_from_op(op) for op in misc_ops),
            "count": len(misc_ops)
        }
    }
    
    # Add dense layers
    for layer_idx, layer_ops in enumerate(dense_layers):
        layer_data = {
            "layer_index": layer_idx,
            "operations": layer_ops,
            "total_flops": sum(get_flops_from_op(op) for op in layer_ops),
            "total_expected_flops": sum(get_expected_flops_from_op(op) for op in layer_ops),
            "total_flops_per_token": sum(get_expected_flops_per_token_from_op(op) for op in layer_ops),
            "count": len(layer_ops)
        }
        output_data["dense_layers"].append(layer_data)
    
    # Add expert layers
    for layer_idx, layer_ops in enumerate(expert_layers):
        layer_data = {
            "layer_index": layer_idx,
            "operations": layer_ops,
            "total_flops": sum(get_flops_from_op(op) for op in layer_ops),
            "total_expected_flops": sum(get_expected_flops_from_op(op) for op in layer_ops),
            "total_flops_per_token": sum(get_expected_flops_per_token_from_op(op) for op in layer_ops),
            "count": len(layer_ops)
        }
        output_data["expert_layers"].append(layer_data)
    
    # Write to file
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Layer operations written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

def get_expected_flops_from_op(op: Dict) -> float:
    """Extract expected sharded FLOPs from operation."""
    try:
        expected_flops_str = op.get('expected_sharded_flops', '')
        if expected_flops_str:
            return float(expected_flops_str)
    except (ValueError, TypeError):
        pass
    return 0.0

def get_expected_flops_per_token_from_op(op: Dict) -> float:
    """Extract expected FLOPs per token from operation."""
    try:
        expected_per_token_str = op.get('expected_flops_per_token', '')
        if expected_per_token_str:
            return float(expected_per_token_str)
    except (ValueError, TypeError):
        pass
    return 0.0

def get_expected_unsharded_flops_from_op(op: Dict) -> float:
    """Extract expected unsharded FLOPs from operation."""
    try:
        expected_unsharded_str = op.get('expected_unsharded_flops', '')
        if expected_unsharded_str:
            return float(expected_unsharded_str)
    except (ValueError, TypeError):
        pass
    return 0.0

def calculate_discrepancy(measured: float, expected: float) -> Tuple[float, float]:
    """
    Calculate discrepancy between measured and expected FLOPs.
    
    Returns:
        (absolute_discrepancy, relative_discrepancy_ratio)
        - absolute_discrepancy: expected - measured
        - relative_discrepancy_ratio: (expected - measured) / expected if expected > 0, else 0
    """
    if expected == 0:
        return 0.0, 0.0
    absolute = expected - measured
    relative = absolute / expected
    return absolute, relative

def print_layer_operations(layer_idx: int, operations: List[Dict], layer_type: str):
    """Print all operations for a layer with expected FLOPs and discrepancy."""
    print("=" * 100)
    print(f"{layer_type.upper()} LAYER {layer_idx} - OPERATIONS")
    print("=" * 100)
    print()
    
    if not operations:
        print(f"No operations found for {layer_type} layer {layer_idx}.")
        return
    
    total_flops = sum(get_flops_from_op(op) for op in operations)
    total_expected_flops = sum(get_expected_flops_from_op(op) for op in operations)
    total_abs_discrepancy, total_rel_discrepancy = calculate_discrepancy(total_flops, total_expected_flops)
    
    print(f"Total operations: {len(operations)}")
    print(f"Total Measured FLOPs: {total_flops:.2e}")
    print(f"Total Expected FLOPs: {total_expected_flops:.2e}")
    print(f"Total Discrepancy: {total_abs_discrepancy:.2e} ({total_rel_discrepancy*100:.2f}%)")
    print()
    
    # Print header
    print(f"{'Idx':<10} {'op_name':<60} {'Measured':<15} {'Expected':<15} {'Discrepancy':<15} {'Ratio':<10}")
    print("-" * 125)
    
    for op in operations:
        idx = op.get('idx', 'N/A')
        op_name = op.get('op_name', '(empty)')
        # Truncate long op_names for display
        if len(op_name) > 58:
            op_name = op_name[:55] + "..."
        
        measured_flops = get_flops_from_op(op)
        expected_flops = get_expected_flops_from_op(op)
        abs_discrepancy, rel_discrepancy = calculate_discrepancy(measured_flops, expected_flops)
        
        # Format ratio as percentage
        ratio_str = f"{rel_discrepancy*100:.2f}%" if expected_flops > 0 else "N/A"
        
        print(f"{idx:<10} {op_name:<60} {measured_flops:<15.2e} {expected_flops:<15.2e} {abs_discrepancy:<15.2e} {ratio_str:<10}")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Count operations by op_name from CSV file'
    )
    parser.add_argument('csv_file', help='Path to comp_stats.csv or comm_stats.csv file')
    parser.add_argument('--dense-layers', type=int, help='Number of dense layers')
    parser.add_argument('--expert-layers', type=int, help='Number of expert layers')
    parser.add_argument('--output', type=str, help='Output JSON file path to write all layer operations')
    parser.add_argument('--flops-results', type=str, help='Path to flops_results.csv file with expected FLOPs')
    
    args = parser.parse_args()
    
    # Parse CSV
    print(f"Parsing CSV file: {args.csv_file}")
    operations = parse_csv(args.csv_file)
    print(f"Found {len(operations)} total operations")
    print()
    
    # Parse flops_results.csv if provided
    lookup_by_scope_einsum = {}
    lookup_by_op_name = {}
    if args.flops_results:
        print(f"Parsing flops results file: {args.flops_results}")
        flops_results = parse_flops_results_csv(args.flops_results)
        if flops_results:
            lookup_by_scope_einsum, lookup_by_op_name = build_flops_lookup(flops_results)
            print(f"Loaded {len(lookup_by_scope_einsum)} flops result entries (by scope/einsum)")
            print(f"Loaded {len(lookup_by_op_name)} flops result entries (by op_name)")
            # Enrich all operations with expected FLOPs
            matched, total, unmatched_ops = enrich_operations_with_flops(
                operations, lookup_by_scope_einsum, lookup_by_op_name
            )
            print(f"Matched {matched}/{total} operations to flops results")
            if unmatched_ops:
                print(f"\nWarning: {len(unmatched_ops)} operations could not be matched:")
                for op_name in unmatched_ops[:10]:  # Show first 10
                    print(f"  - {op_name}")
                if len(unmatched_ops) > 10:
                    print(f"  ... and {len(unmatched_ops) - 10} more")
        print()
    
    # Count by op_name (always show this)
    counts = count_by_op_name(operations)
    print_op_name_counts(counts)
    
    # If layer counts provided, assign operations to layers
    if args.dense_layers is not None and args.expert_layers is not None:
        num_dense = args.dense_layers
        num_expert = args.expert_layers
        
        print("=" * 100)
        print("ASSIGNING OPERATIONS TO LAYERS")
        print("=" * 100)
        print()
        print(f"Dense layers: {num_dense}")
        print(f"Expert layers: {num_expert}")
        print()
        
        dense_layers, expert_layers, misc_ops = assign_operations_to_layers(
            operations, num_dense, num_expert
        )
        
        print("=" * 100)
        print("LAYER ASSIGNMENT SUMMARY")
        print("=" * 100)
        print()
        
        print(f"Dense layers:")
        dense_total_flops = 0.0
        dense_total_expected_flops = 0.0
        dense_total_expected_flops_per_token = 0.0
        for i, layer_ops in enumerate(dense_layers):
            layer_flops = sum(get_flops_from_op(op) for op in layer_ops)
            layer_expected_flops = sum(get_expected_flops_from_op(op) for op in layer_ops)
            layer_expected_flops_per_token = sum(get_expected_flops_per_token_from_op(op) for op in layer_ops)
            dense_total_flops += layer_flops
            dense_total_expected_flops += layer_expected_flops
            dense_total_expected_flops_per_token += layer_expected_flops_per_token
            layer_abs_disc, layer_rel_disc = calculate_discrepancy(layer_flops, layer_expected_flops)
            print(f"  Layer {i}: {len(layer_ops)} operations, Measured: {layer_flops:.2e}, Expected: {layer_expected_flops:.2e}, Discrepancy: {layer_abs_disc:.2e} ({layer_rel_disc*100:.2f}%), Expected FLOPs/token: {layer_expected_flops_per_token:.2e}")
        dense_abs_disc, dense_rel_disc = calculate_discrepancy(dense_total_flops, dense_total_expected_flops)
        print(f"  Total dense layers - Measured: {dense_total_flops:.2e}, Expected: {dense_total_expected_flops:.2e}, Discrepancy: {dense_abs_disc:.2e} ({dense_rel_disc*100:.2f}%), Expected FLOPs/token: {dense_total_expected_flops_per_token:.2e}")
        print()
        
        print(f"Expert layers:")
        expert_total_flops = 0.0
        expert_total_expected_flops = 0.0
        expert_total_expected_flops_per_token = 0.0
        for i, layer_ops in enumerate(expert_layers):
            layer_flops = sum(get_flops_from_op(op) for op in layer_ops)
            layer_expected_flops = sum(get_expected_flops_from_op(op) for op in layer_ops)
            layer_expected_flops_per_token = sum(get_expected_flops_per_token_from_op(op) for op in layer_ops)
            expert_total_flops += layer_flops
            expert_total_expected_flops += layer_expected_flops
            expert_total_expected_flops_per_token += layer_expected_flops_per_token
            layer_abs_disc, layer_rel_disc = calculate_discrepancy(layer_flops, layer_expected_flops)
            print(f"  Layer {i}: {len(layer_ops)} operations, Measured: {layer_flops:.2e}, Expected: {layer_expected_flops:.2e}, Discrepancy: {layer_abs_disc:.2e} ({layer_rel_disc*100:.2f}%), Expected FLOPs/token: {layer_expected_flops_per_token:.2e}")
        expert_abs_disc, expert_rel_disc = calculate_discrepancy(expert_total_flops, expert_total_expected_flops)
        print(f"  Total expert layers - Measured: {expert_total_flops:.2e}, Expected: {expert_total_expected_flops:.2e}, Discrepancy: {expert_abs_disc:.2e} ({expert_rel_disc*100:.2f}%), Expected FLOPs/token: {expert_total_expected_flops_per_token:.2e}")
        print()
        
        misc_total_flops = sum(get_flops_from_op(op) for op in misc_ops)
        misc_total_expected_flops = sum(get_expected_flops_from_op(op) for op in misc_ops)
        misc_total_expected_flops_per_token = sum(get_expected_flops_per_token_from_op(op) for op in misc_ops)
        misc_abs_disc, misc_rel_disc = calculate_discrepancy(misc_total_flops, misc_total_expected_flops)
        print(f"Misc operations (unassigned): {len(misc_ops)}, Measured: {misc_total_flops:.2e}, Expected: {misc_total_expected_flops:.2e}, Discrepancy: {misc_abs_disc:.2e} ({misc_rel_disc*100:.2f}%), Expected FLOPs/token: {misc_total_expected_flops_per_token:.2e}")
        print()
        
        # Calculate grand total
        grand_total_flops = dense_total_flops + expert_total_flops + misc_total_flops
        grand_total_expected_flops = dense_total_expected_flops + expert_total_expected_flops + misc_total_expected_flops
        grand_total_expected_flops_per_token = dense_total_expected_flops_per_token + expert_total_expected_flops_per_token + misc_total_expected_flops_per_token
        grand_abs_disc, grand_rel_disc = calculate_discrepancy(grand_total_flops, grand_total_expected_flops)
        print(f"Grand Total - Measured: {grand_total_flops:.2e}, Expected: {grand_total_expected_flops:.2e}, Discrepancy: {grand_abs_disc:.2e} ({grand_rel_disc*100:.2f}%)")
        print()
        print("=" * 100)
        print(f"TOTAL EXPECTED FLOPS PER TOKEN (sum across all layers): {grand_total_expected_flops_per_token:.2e}")
        print("=" * 100)
        print()
        
        # Print first dense layer
        if dense_layers:
            print_layer_operations(0, dense_layers[0], "dense")
        
        # Print first expert layer
        if expert_layers:
            print_layer_operations(0, expert_layers[0], "expert")
        
        # Print misc operations
        if misc_ops:
            misc_total_flops = sum(get_flops_from_op(op) for op in misc_ops)
            misc_total_expected_flops = sum(get_expected_flops_from_op(op) for op in misc_ops)
            misc_abs_discrepancy, misc_rel_discrepancy = calculate_discrepancy(misc_total_flops, misc_total_expected_flops)
            
            print("=" * 100)
            print("MISC OPERATIONS (UNASSIGNED)")
            print("=" * 100)
            print()
            print(f"Total misc operations: {len(misc_ops)}")
            print(f"Total Measured FLOPs: {misc_total_flops:.2e}")
            print(f"Total Expected FLOPs: {misc_total_expected_flops:.2e}")
            print(f"Total Discrepancy: {misc_abs_discrepancy:.2e} ({misc_rel_discrepancy*100:.2f}%)")
            print()
            print(f"{'Idx':<10} {'op_name':<60} {'Measured':<15} {'Expected':<15} {'Discrepancy':<15} {'Ratio':<10}")
            print("-" * 125)
            
            for op in misc_ops:
                idx = op.get('idx', 'N/A')
                op_name = op.get('op_name', '(empty)')
                # Truncate long op_names for display
                if len(op_name) > 58:
                    op_name = op_name[:55] + "..."
                
                measured_flops = get_flops_from_op(op)
                expected_flops = get_expected_flops_from_op(op)
                abs_discrepancy, rel_discrepancy = calculate_discrepancy(measured_flops, expected_flops)
                
                # Format ratio as percentage
                ratio_str = f"{rel_discrepancy*100:.2f}%" if expected_flops > 0 else "N/A"
                
                print(f"{idx:<10} {op_name:<60} {measured_flops:<15.2e} {expected_flops:<15.2e} {abs_discrepancy:<15.2e} {ratio_str:<10}")
            
            print()
        
        # Calculate totals across all operations for final summary
        all_operations = []
        for layer_ops in dense_layers:
            all_operations.extend(layer_ops)
        for layer_ops in expert_layers:
            all_operations.extend(layer_ops)
        all_operations.extend(misc_ops)
        
        total_unsharded_flops_expected = sum(get_expected_unsharded_flops_from_op(op) for op in all_operations)
        total_sharded_flops_expected = sum(get_expected_flops_from_op(op) for op in all_operations)
        total_sharded_flops_measured = sum(get_flops_from_op(op) for op in all_operations)
        total_flops_per_token = sum(get_expected_flops_per_token_from_op(op) for op in all_operations)
        
        print("=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)
        print()
        print(f"1. Total Unsharded FLOPs Expected: {total_unsharded_flops_expected:.2e}")
        print(f"2. Total Sharded FLOPs Expected:   {total_sharded_flops_expected:.2e}")
        print(f"3. Total Sharded FLOPs Measured:   {total_sharded_flops_measured:.2e}")
        print(f"4. Total FLOPs Per Token:          {total_flops_per_token:.2e}")
        print()
        print("=" * 100)
        
        # Write to output file if requested
        if args.output:
            write_layer_operations_to_file(dense_layers, expert_layers, misc_ops, args.output)

if __name__ == '__main__':
    main()
