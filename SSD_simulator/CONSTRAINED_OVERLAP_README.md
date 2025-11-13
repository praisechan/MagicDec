# Constrained Head Overlap Latency Feature

## Overview

Added a new latency calculation method that considers constraints on the number of heads that can execute concurrently across all planes. This addresses scenarios where hardware limitations (memory bandwidth, execution units, power) prevent all heads from running in parallel.

## Changes Made

### 1. New Function: `get_constrained_overlap_latency()`

**Location:** `simulator_for_head_overlap_number.py` (after `get_plane_reads_per_head()`)

**Purpose:** Calculate total latency when at most N heads can execute concurrently.

**Algorithm:**
- Calculates each head's latency (max reads across its planes)
- Schedules heads sequentially, respecting the concurrency limit
- When the limit is reached, waits for the earliest-finishing head to complete
- Returns total latency and detailed schedule for each head

**Parameters:**
- `reads_per_head`: List[List[int]] - reads per plane for each head
- `max_concurrent_heads`: int - maximum number of concurrent heads

**Returns:**
- `total_latency`: int - total cycles needed
- `head_schedules`: List[Tuple] - (head_idx, start_cycle, end_cycle) for each head

### 2. New Command-Line Argument

**Argument:** `--max_concurrent_heads`
- Type: int
- Default: None (no constraint)
- Description: Maximum number of heads that can execute concurrently

### 3. Integration into Main Simulation

**Changes in `main()` function:**

1. Added `step_constrained_overlap_latency` tracking variable
2. Calculates constrained latency for each layer when `--max_concurrent_heads` is specified
3. Prints constrained latency for each step
4. Accumulates total constrained latency across all steps

**CSV Output Updates:**
- Added `max_concurrent_heads` column in header
- Added `step_constrained_overlap_latency` column in header
- Values are "N/A" when constraint is not applied

## Usage Examples

### Basic Usage
```bash
python simulator_for_head_overlap_number.py \
  --max_concurrent_heads 4 \
  --num_channels 1 \
  --chips_per_channel 1 \
  --dies_per_chip 1 \
  --planes_per_die 8 \
  --page_size_bytes 16384 \
  --profiling_dir ./profiling_data \
  --model_name llama \
  --dataset hotpot \
  --generate_name speculate_0_0 \
  --budget_ratio 0.25 \
  --cluster_size 16 \
  --num_replica 4 \
  --max_latency_calculate
```

### Test the Feature
```bash
cd /home/juchanlee/MagicDec/SSD_simulator
python test_constrained_overlap.py
```

This runs a demonstration with 8 heads and shows how latency changes with different concurrency limits (1, 2, 4, 8, unlimited).

## Expected Results

### Latency Comparison (8 heads example):

| Constraint | Total Latency | Explanation |
|-----------|---------------|-------------|
| max=1 | 1005 cycles | Sequential execution (sum of all head latencies) |
| max=2 | 555 cycles | Only 2 heads overlap at any time |
| max=4 | 310 cycles | Up to 4 heads overlap |
| max=8 | 160 cycles | All heads can run in parallel |
| Unlimited | 160 cycles | Same as max ≥ num_heads |

### Output Format

**Console Output (per step):**
```
Step 0 - total latency: 12345
Step 0 - total latency(head overlap): 6789
Step 0 - ideal total latency: 5432
Step 0 - constrained overlap latency (max 4 concurrent heads): 8901
Step 0 - total max head latency: 2345
```

**CSV Columns Added:**
- `max_concurrent_heads`: The constraint value (or "N/A")
- `step_constrained_overlap_latency`: Latency with constraint (or "N/A")

## Use Cases

This feature is valuable for:

1. **Memory Bandwidth Constraints**: When memory system can only support N concurrent head reads
2. **Hardware Parallelism Limits**: When hardware has limited parallel execution units
3. **Power Budget Constraints**: When power limits restrict concurrent operations
4. **Realistic Performance Modeling**: More accurate latency predictions for real hardware
5. **Design Space Exploration**: Understanding the impact of parallelism constraints on performance

## Implementation Notes

### Key Design Decisions:

1. **Greedy Scheduling**: Heads are scheduled in order, waiting only when necessary
2. **Conservative Estimate**: Assumes heads cannot be preempted or reordered
3. **Plane-Level Granularity**: Each head's latency is the max across its planes
4. **Backward Compatible**: Setting `--max_concurrent_heads` to None or omitting it maintains original behavior

### Differences from Existing Metrics:

- **`step_total_latency`**: Sum of all head latencies (no overlap)
- **`step_overlap_latency`**: Max of summed plane reads (perfect overlap)
- **`step_ideal_latency`**: Average latency assuming perfect load balancing
- **`step_constrained_overlap_latency`**: Realistic latency with concurrency limit ✨ NEW

## Testing

Run the test script to verify the algorithm:
```bash
python /home/juchanlee/MagicDec/SSD_simulator/test_constrained_overlap.py
```

Expected output shows latency calculations for different concurrency constraints with detailed scheduling information.

## Future Enhancements

Potential improvements:
1. More sophisticated scheduling algorithms (e.g., shortest-job-first)
2. Dynamic concurrency limits based on head characteristics
3. Per-plane concurrency constraints
4. Visualization of head schedules
5. Support for head preemption and reordering
