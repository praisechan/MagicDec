#!/usr/bin/env python
"""
Test script to demonstrate the constrained overlap latency calculation.
This shows how limiting concurrent head execution affects total latency.
"""

from typing import List

def get_constrained_overlap_latency(reads_per_head: List[List[int]], max_concurrent_heads: int) -> tuple:
    """
    Calculate latency with constraint on number of concurrent heads.
    
    Args:
        reads_per_head: List of reads per plane for each head. 
                       reads_per_head[head_idx][plane_idx] = number of reads
        max_concurrent_heads: Maximum number of heads that can execute concurrently
    
    Returns:
        tuple: (total_latency, head_schedules)
            - total_latency: Total cycles needed with overlap constraint
            - head_schedules: List of (head_idx, start_cycle, end_cycle) tuples
    """
    num_heads = len(reads_per_head)
    num_planes = len(reads_per_head[0]) if num_heads > 0 else 0
    
    # Calculate the latency (max reads across planes) for each head
    head_latencies = [max(plane_reads) for plane_reads in reads_per_head]
    
    # Track when each head finishes
    head_schedules = []
    current_cycle = 0
    active_heads = []  # List of (head_idx, end_cycle)
    
    for head_idx in range(num_heads):
        # Remove finished heads from active list
        active_heads = [(h, end) for h, end in active_heads if end > current_cycle]
        
        # If we have max_concurrent_heads active, wait for one to finish
        if len(active_heads) >= max_concurrent_heads:
            # Find the earliest finish time among active heads
            earliest_finish = min(end for _, end in active_heads)
            current_cycle = earliest_finish
            # Remove heads that finished
            active_heads = [(h, end) for h, end in active_heads if end > current_cycle]
        
        # Schedule current head
        start_cycle = current_cycle
        end_cycle = start_cycle + head_latencies[head_idx]
        head_schedules.append((head_idx, start_cycle, end_cycle))
        active_heads.append((head_idx, end_cycle))
        
    # Total latency is when the last head finishes
    total_latency = max(end for _, _, end in head_schedules)
    
    return total_latency, head_schedules

# Example: 8 heads with different read patterns across 4 planes
# Each inner list represents reads per plane for that head
reads_per_head = [
    [100, 120, 110, 130],  # Head 0
    [90, 95, 100, 105],     # Head 1
    [150, 140, 160, 145],   # Head 2
    [80, 85, 90, 95],       # Head 3
    [110, 115, 105, 120],   # Head 4
    [130, 125, 135, 140],   # Head 5
    [95, 100, 90, 105],     # Head 6
    [140, 135, 145, 150],   # Head 7
]

print("=" * 80)
print("CONSTRAINED OVERLAP LATENCY DEMONSTRATION")
print("=" * 80)
print(f"\nNumber of heads: {len(reads_per_head)}")
print(f"Number of planes: {len(reads_per_head[0])}")
print("\nReads per plane for each head:")
for i, reads in enumerate(reads_per_head):
    max_reads = max(reads)
    print(f"  Head {i}: {reads} -> max latency = {max_reads}")

print("\n" + "=" * 80)
print("LATENCY CALCULATIONS WITH DIFFERENT CONCURRENCY CONSTRAINTS")
print("=" * 80)

# Test different concurrency limits
for max_concurrent in [1, 2, 4, 8, None]:
    if max_concurrent is None:
        # Unlimited concurrency (all heads can run simultaneously)
        unconstrained_latency = max(max(reads) for reads in reads_per_head)
        print(f"\nUnlimited concurrency (all heads parallel):")
        print(f"  Total latency: {unconstrained_latency} cycles")
        print(f"  (Simply the maximum latency across all heads)")
    else:
        latency, schedules = get_constrained_overlap_latency(reads_per_head, max_concurrent)
        print(f"\nMax {max_concurrent} concurrent head(s):")
        print(f"  Total latency: {latency} cycles")
        print(f"  Head schedules:")
        for head_idx, start, end in schedules:
            duration = end - start
            print(f"    Head {head_idx}: cycles [{start:4d} - {end:4d}] (duration: {duration} cycles)")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("\nKey observations:")
print("1. With max_concurrent_heads=1: Heads execute sequentially (highest latency)")
print("2. With max_concurrent_heads=2: Only 2 heads can overlap at any time")
print("3. With max_concurrent_heads=4: Up to 4 heads can overlap")
print("4. With max_concurrent_heads=8: All 8 heads can run in parallel")
print("5. Unlimited: Same as max_concurrent_heads >= num_heads")
print("\nThis constraint is useful when:")
print("- Memory bandwidth limits how many heads can read simultaneously")
print("- Hardware has limited parallel execution units")
print("- Power constraints limit concurrent operations")
